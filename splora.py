# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy
import torch
from transformers import AutoModelForCausalLM
from .similarity import compute_diem_torch, compute_diem_torch_iqr
from .similarity import compute_diem,mean_std_entropy,median_absolute_deviation
from .similarity import soft_cosine_similarity,spectral_metrics
from .similarity import ica_transform,compute_cosine_similarity
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize



class SPLoRA:
    def __init__(self, peft_model:torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SPLoRA:
        path = './LLM_Models/llama-2-7b-chat/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'ft_path_name',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat/' #you should modify the path

        SPlora = SPLoRA(pmodel, SPLoRAConfig)

        Finally, you can get the projected model by "splora.model".
        """
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        print("self.peft_config", self.peft_config)
        self.model_ori = copy.deepcopy(peft_model)
        project_matrix = self.get_aligned_matrix()

        if self.config.select_layers_type == 'threshold':
            self.model, _ = self.projected_weighted(project_matrix, self.config.threshold, show_info=True)

        elif self.config.select_layers_type == 'number':
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            
            thrs = numpy.sort(cos)[::-1][:self.config.num_proj_layers][-1] 
            self.model, _ = self.projected_weighted(project_matrix, thrs, show_info=True)
        else:
            raise ValueError("The method of select_layer_type should be threshold or number.")

    def get_aligned_matrix(self):
        """
        Get projected matrix by following the config (target_modules) from the peft model.
        The dimensions between the base model's weights and the aligned model's weights should be the same.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        v = []
        proj_modules = list(self.peft_config.target_modules)
        print("proj_modules",proj_modules)

        for (b_name, b_param) , (a_name, a_param) in zip (base_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules):
                
                # print('b_name',b_name, a_name) 
                # print("b_param", b_param.shape, a_param.shape)[4096, 4096]
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(self.config.devices)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())
        print("v final", len(v)) 
        return v

    def projected_weighted(self, project_matrix, thrs_cos, show_info=False):
        v = project_matrix
        # print("v", len(v)) 64
        idx = 0
        i = 0
        dis = []
        cos_total = []
        for (name, param),(name_ori, param_ori) in zip(self.peft_model.named_parameters(), self.model_ori.named_parameters()):
            if 'lora' in name:
                if param.shape[0] == self.peft_config.r:
                   
                    B = copy.deepcopy(param_ori)
                    
                if param.shape[0] != self.peft_config.r:
                    P = v[idx].to(param.device)
                    
                    W = torch.mm(P, param_ori.data)
                   
                    fW = torch.mm(W, B) 
                   
                    ori = torch.mm(param_ori, B)
                   
                    W_new = torch.mm(P, param_ori.data)
                    
                    weight_samples = [(fW,ori) for _ in range(50)]
                    ediem = compute_diem_torch(fW,ori,weight_samples=weight_samples)
                    ediem_iqr = compute_diem_torch_iqr(fW,ori,weight_samples=weight_samples)

                    cos = ediem_iqr
                    cos_total.append(cos)

                    if cos <=  thrs_cos:
                        i+=1
                        # param.data =  W_new
                        param.data = torch.zeros(4096, 8).to(param.device)
                    else:
                        param.data = param_ori
                    dist = 1 / (1+torch.norm(param.data.reshape(1,-1)-W.reshape(1,-1)))
                    # print("dist", dist)

                    dis.append(dist.item())
                    idx += 1
        if show_info:
            
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}")
        return self.peft_model, cos_total

