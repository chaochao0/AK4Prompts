import torch
import numpy as np
from keywords.keywords_table import KeywordsTable,Config
class AK4PromptsPipeline():
    def __init__(self,pipeline=None,ak4prompts=None,keywords_filename="keywords_list.txt"):
        super().__init__()
        self.pipeline = pipeline
        self.ak4prompts = ak4prompts
        self.ak4prompts.eval()

        self.device = next(ak4prompts.parameters()).device

        keywords_table = KeywordsTable(Config(device='cpu',keywords_filename=keywords_filename))

        self.keywords_embs = torch.from_numpy(np.array(keywords_table.flavors.embeds)).unsqueeze(dim=0).to(self.device)
        self.labels = keywords_table.flavors.labels

    def keywords_ranking(self,prompt,scores_weights,topk=10):
        
        prompt_inputs = self.pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.pipeline.tokenizer.model_max_length,
        )
        prompt_ids = prompt_inputs.input_ids.to(self.device)
        prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]
        attention_mask = (prompt_inputs.attention_mask==0).to(self.device)
        with torch.no_grad():
            aesthetic_scores,clip_scores,hps_scores = self.ak4prompts.forward(prompt_embeds.to(torch.float32), attention_mask, self.keywords_embs.to(torch.float32))

        avg_score = aesthetic_scores*scores_weights['aesthetic'] + clip_scores*scores_weights['clip'] + hps_scores*scores_weights['hps']
        k_values, k_indices = torch.topk(avg_score, k=topk) 
        k_indices.unsqueeze(dim=0) 
        k_values.unsqueeze(dim=0)

        append_keywords_choosed = [","+",".join([self.labels[idx] for idx in indices]) for indices in k_indices]
        prompts_with_keywords = [prompt + append_keywords for prompt, append_keywords in zip(prompt,append_keywords_choosed)]
        return prompts_with_keywords
