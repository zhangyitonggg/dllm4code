# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Import the generate function required for LLaDA models
from repoqa.provider.llada_generate import generate

from repoqa.provider.base import BaseProvider
from repoqa.provider.request import construct_message_list, hacky_assistant_stop_seq


class HfProvider(BaseProvider):
    def __init__(self, model, trust_remote_code=False, attn_implementation=None):
        self.model_name = model
        self.trust_remote_code = trust_remote_code
        
        # Detect model type
        self.is_diffusion_model = any(model_name in model for model_name in [
            "Dream-Coder-v0-Instruct-7B", 
            "DiffuCoder-7B-cpGRPO", 
            "Dream-v0-Instruct-7B",
            "DreamOn"
        ])
        
        # Detect LLaDA models
        self.is_llada_model = "LLaDA" in model
        
        # Mark DreamOn model separately for special handling
        self.is_DreamOn_model = "DreamOn" in model
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        
        # Load different models based on model type
        if self.is_diffusion_model or self.is_llada_model:
            # Diffusion models and LLaDA models use AutoModel
            self.hf_model = AutoModel.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            ).to("cuda").eval()
        else:
            # Loading method for regular models
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                torch_dtype="auto",
            ).cuda()
        
        self.stop_seq = []
        if self.tokenizer.chat_template:
            self.stop_seq.append(hacky_assistant_stop_seq(self.tokenizer))
        
        # Set default parameters based on model type
        self._set_default_params()

    def _set_default_params(self):
        """Set default generation parameters based on model type"""
        if "DiffuCoder-7B-cpGRPO" in self.model_name:
            self.default_steps = 256  # 256//1
            self.default_max_new_tokens = 256
            self.default_temperature = 0.1
        elif "Dream-Coder-v0-Instruct-7B" in self.model_name:
            self.default_steps = 256
            self.default_max_new_tokens = 256
            self.default_temperature = 0.1
        elif "Dream-v0-Instruct-7B" in self.model_name:
            self.default_steps = 256
            self.default_max_new_tokens = 256
            self.default_temperature = 0.1
        elif self.is_DreamOn_model:
            # Default parameters for DreamOn model
            self.default_steps = 128
            self.default_max_new_tokens = 128
            self.default_temperature = 0.1
            self.default_mask_length = 128  # Manually added mask length
            self.default_alg = "maskgit_plus"
            self.default_number_transfer_tokens = 1
        elif "LLaDA" in self.model_name:
            # Default parameters for LLaDA model
            self.default_steps = 256
            self.default_gen_length = 256
            self.default_block_length = 16
            self.default_temperature = 0
            self.default_cfg_scale = 0.0
            self.default_remasking = 'low_confidence'
            self.default_max_new_tokens = 256
        else:
            self.default_steps = None
            self.default_max_new_tokens = 512
            self.default_temperature = 0.0

    @torch.inference_mode()
    def generate_reply(
        self, question, n=1, max_tokens=None, temperature=None, system_msg=None
    ) -> List[str]:
        assert temperature != 0 or n == 1, "n must be 1 when temperature is 0"
        
        # Use model default parameters if not specified
        max_tokens = max_tokens or self.default_max_new_tokens
        temperature = temperature or self.default_temperature

        if self.is_DreamOn_model:
            return self._DreamOn_model_generate(
                question, n, max_tokens, temperature, system_msg
            )
        elif self.is_diffusion_model:
            return self._diffusion_model_generate(
                question, n, max_tokens, temperature, system_msg
            )
        elif self.is_llada_model:
            return self._llada_model_generate(
                question, n, max_tokens, temperature, system_msg
            )
        else:
            return self._default_model_generate(
                question, n, max_tokens, temperature, system_msg
            )

    def _DreamOn_model_generate(
        self, question, n=1, max_tokens=None, temperature=None, system_msg=None
    ) -> List[str]:
        """Handle generation logic for DreamOn model; manual mask token addition is required"""
        # Get default parameters
        max_tokens = max_tokens or self.default_max_new_tokens
        temperature = temperature or self.default_temperature
        steps = self.default_steps
        mask_length = self.default_mask_length
        alg = self.default_alg
        number_transfer_tokens = self.default_number_transfer_tokens
        
        # Construct prompt
        prompt = question.strip()
        if system_msg:
            prompt = f"{system_msg}\n{prompt}"
        
        # Process input and add mask tokens
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )
        
        input_ids = inputs.input_ids
        input_ids_len = len(input_ids[0])
        
        # Manually add mask tokens
        mask_tokens = torch.tensor([[self.tokenizer.mask_token_id] * mask_length])
        input_ids = torch.cat((input_ids, mask_tokens), dim=-1).to("cuda")
        
        # Call diffusion_generate method
        output = self.hf_model.diffusion_generate(
            input_ids,
            max_new_tokens=max_tokens,
            output_history=False,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg=alg,
            alg_temp=0.,
            number_transfer_tokens=number_transfer_tokens,
            return_dict_in_generate=True,
        )
        
        # Decode generated content (exclude input part), using the same processing method as DreamCoder
        gen_strs = []
        for seq in output.sequences:
            decoded = self.tokenizer.decode(
                seq[input_ids_len:].tolist(),
                skip_special_tokens=False
            )
            # Same ending processing as DreamCoder
            decoded = decoded.split(self.tokenizer.eos_token)[0]
            gen_strs.append(decoded)
        
        # Duplicate results if multiple return sequences are needed
        if n > 1:
            gen_strs = gen_strs * n
        
        return gen_strs[:n]

    def _diffusion_model_generate(
        self, question, n=1, max_tokens=256, temperature=0.4, system_msg=None
    ) -> List[str]:
        """Handle generation logic for diffusion models (Dream-Coder, DiffuCoder, etc.)"""
        # Construct different prompts based on model type
        if "DiffuCoder-7B-cpGRPO" in self.model_name:
            # DiffuCoder uses Qwen-style template
            system_msg = system_msg or "You are a helpful assistant."
            prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{question.strip()}
<|im_end|>
<|im_start|>assistant
"""
            inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            # Dream series models use chat template
            messages = construct_message_list(question, system_msg)
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            )
        
        input_ids = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda")
        input_length = input_ids.size(-1)
        
        # Set diffusion generation parameters
        steps = self.default_steps or max_tokens
        
        # Call diffusion_generate method
        output = self.hf_model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
        )
        
        # Decode generation results
        gen_strs = []
        for p, g in zip(input_ids, output.sequences):
            decoded = self.tokenizer.decode(g[len(p):].tolist())
            # Process stop tokens based on different models
            if "DiffuCoder-7B-cpGRPO" in self.model_name:
                decoded = decoded.split('<|dlm_pad|>')[0]
            else:
                decoded = decoded.split(self.tokenizer.eos_token)[0]
            gen_strs.append(decoded)
        
        # Currently duplicate simply if multiple return sequences are needed (actual use may require multiple generations)
        if n > 1:
            gen_strs = gen_strs * n
        print(gen_strs)
        return gen_strs[:n]

    def _llada_model_generate(
        self, question, n=1, max_tokens=None, temperature=None, system_msg=None
    ) -> List[str]:
        """Handle generation logic for LLaDA models"""
        # Construct conversation messages
        messages = construct_message_list(question, system_msg)
        user_input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        print(user_input)
        input_ids = self.tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to("cuda").unsqueeze(0)
        
        # Set generation parameters
        steps = self.default_steps
        gen_length = max_tokens or self.default_gen_length
        block_length = self.default_block_length
        temperature = temperature or self.default_temperature
        cfg_scale = self.default_cfg_scale
        remasking = self.default_remasking
        
        # Call LLaDA-specific generate function
        out = generate(
            self.hf_model, 
            input_ids, 
            steps=steps, 
            gen_length=gen_length, 
            block_length=block_length, 
            temperature=temperature, 
            cfg_scale=cfg_scale, 
            remasking=remasking
        )
        
        # Decode generation results, find and truncate to the first end marker
        end_markers = ["<|endoftext|>", "<|eot_id|>"]
        gen_strs = []
        for output in out:
            # Retain special tokens during decoding
            decoded = self.tokenizer.decode(
                output[input_ids.size(1):], 
                skip_special_tokens=False  # Do not automatically skip special tokens
            )
            
            # Find the position of the first end marker
            min_pos = None
            for marker in end_markers:
                pos = decoded.find(marker)
                if pos != -1 and (min_pos is None or pos < min_pos):
                    min_pos = pos
            
            # Truncate text based on the found position
            if min_pos is not None:
                decoded = decoded[:min_pos]
            
            gen_strs.append(decoded)
        
        # Duplicate results if multiple return sequences are needed
        if n > 1:
            gen_strs = gen_strs * n
        print(gen_strs)
        return gen_strs[:n]

    def _default_model_generate(
        self, question, n=1, max_tokens=1024, temperature=0.0, system_msg=None
    ) -> List[str]:
        """Default model generation logic"""
        if "Qwen3" in self.model_name:
            prompt_tokens = self.tokenizer.apply_chat_template(
                construct_message_list(question, system_msg),
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False
            ).cuda()
        else:
            prompt_tokens = self.tokenizer.apply_chat_template(
                construct_message_list(question, system_msg),
                return_tensors="pt",
                add_generation_prompt=True,
            ).cuda()
        input_length = prompt_tokens.size(-1)

        gen_args = {"do_sample": False}
        if temperature > 0:
            gen_args["do_sample"] = True
            gen_args["temperature"] = temperature

        output_text = self.hf_model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            stop_strings=self.stop_seq,
            tokenizer=self.tokenizer,** gen_args,
        )

        gen_strs = [
            self.tokenizer.decode(
                x[input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for x in output_text
        ]
        # print(gen_strs)
        return gen_strs