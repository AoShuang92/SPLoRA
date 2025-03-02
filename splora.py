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



class SafeLoRA:
    def __init__(self, peft_model:torch.nn.Module, config):
        """
        Please use safelora.model to get the projected model.

        How to use SafeLoRA:
        path = './LLM_Models/llama-2-7b-chat-fp16/' # load your base model of the peft model
        model = AutoModelForCausalLM.from_pretrained(path)
        pmodel = PeftModel.from_pretrained(model, 'finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42/',torch_dtype=torch.float16) #load peft model

        SafeLoRAConfig.base_model_path = './LLM_Models/llama-2-7b-hf/'  #you should modify the path
        SafeLoRAConfig.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/' #you should modify the path

        safelora = SafeLoRA(pmodel, SafeLoRAConfig)

        Finally, you can get the projected model by "safelora.model".
        """
        super().__init__()
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config["default"]
        print("self.peft_config", self.peft_config)
        self.model_ori = copy.deepcopy(peft_model)
        project_matrix = self.get_aligned_matrix()
        # print('project_matrix0000', len(project_matrix), project_matrix[0].shape)64, ([4096, 4096])
        print("self.config.select_layers_type", self.config.select_layers_type)

        if self.config.select_layers_type == 'threshold':
            print("threshold", self.config.select_layers_type)
            self.model, _ = self.projected_weighted(project_matrix, self.config.threshold, show_info=True)

        elif self.config.select_layers_type == 'number':
            model, cos = self.projected_weighted(project_matrix, 0.3, show_info=False)
            # print('cos', cos, len(cos)) 64
            print("cos", cos)
            thrs = numpy.sort(cos)[:self.config.num_proj_layers][-1]
            # thrs = numpy.sort(cos)[:5][-1]
            # thrs = numpy.sort(cos)[::-1][:self.config.num_proj_layers][-1] 
            print("thrs", thrs) 
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
                # print("vec", vec.shape) ([4096, 4096])
                # print("vec", vec) all 0
                vec = vec.to(self.config.devices)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                # print("vec", vec.shape)([4096, 4096])
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
                # print("if lora", param.shape[0], self.peft_config.r) 8 8 / 4096 8
                if param.shape[0] == self.peft_config.r:
                    # print("shape equal")
                    B = copy.deepcopy(param_ori)
                    # print("B", B.shape) [8, 4096]
                if param.shape[0] != self.peft_config.r:
                    # print("shape NOT equal")
                    # print("v[idx]", v[idx].shape)
                    P = v[idx].to(param.device)
                    # print("P", P.shape) ([4096, 4096])
                    # print("param_ori.data", param_ori.data.shape) [4096, 8]
                    W = torch.mm(P, param_ori.data)
                    # print("W", W.shape) ([4096, 8])
                    fW = torch.mm(W, B) 
                    # print("FW", fW.shape)([4096, 4096])
                    # print("param_ori", param_ori.shape)[4096, 8]
                    ori = torch.mm(param_ori, B)
                    # print("ori", ori.shape) ([4096, 4096])
                    W_new = torch.mm(P, param_ori.data)
                    # print("W_new", W_new.shape) ([4096, 8])
                    # print("W_new", W_new.shape) [2048, 8]
                    
                    # cos = numpy.round(torch.nn.functional.cosine_similarity(fW.reshape(1,-1), ori.reshape(1,-1)).item(),5)
                    weight_samples = [(fW,ori) for _ in range(50)]
                    ediem = compute_diem_torch(fW,ori,weight_samples=weight_samples)
                    ediem_iqr = compute_diem_torch_iqr(fW,ori,weight_samples=weight_samples)
                    # diem = compute_diem_torch(fW,ori,weight_samples=None)

                    
                    # diem_values = compute_diem(fW,ori)
                    # # print("diem_values", diem_values.shape) [4096]
                    # mad_diem = median_absolute_deviation(diem_values)
                    # mean_diem, std_diem, entropy_diem = mean_std_entropy(diem_values)

                    # feature_similarity = torch.eye(4096).to(param.device)
                    # soft_cosine_sim = soft_cosine_similarity(fW, ori, feature_similarity)
                    # largest_eig, trace_lap, frob_norm = spectral_metrics(soft_cosine_sim)

                    cos = ediem_iqr

                    # # Apply ICA transformation
                    # n_components = 64  # Number of components to extract
                    
                    # fW_ica = ica_transform(fW.cpu() , n_components)
                    # ori_ica = ica_transform(ori.cpu() , n_components)

                    # # Normalize the ICA-transformed embeddings
                    # fW_ica_normalized = normalize(fW_ica, axis=1)
                    # ori_ica_normalized = normalize(ori_ica, axis=1)

                    # # Compute cosine similarity between corresponding rows
                    # cosine_similarities = numpy.array([compute_cosine_similarity(fW_ica_normalized[i], ori_ica_normalized[i]) for i in range(fW_ica_normalized.shape[0])])

                    # # Aggregate the cosine similarities to obtain a single similarity measure
                    # average_cosine_similarity = numpy.mean(cosine_similarities)
                    # cos = average_cosine_similarity
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
            print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {numpy.mean(dis)} (> 0.8 is better).")
            # print(f"{i} layers are projected, cosine threshold is {thrs_cos}")
        return self.peft_model, cos_total

