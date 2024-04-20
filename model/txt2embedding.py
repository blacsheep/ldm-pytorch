import torch.nn
from transformers import BertTokenizerFast, BertModel

# https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/encoders/modules.py#L53
class BERTEmbedding(torch.nn.Module):
    """ Uses a pretrained BERT tokenizer by huggingface."""
    def __init__(self, max_length=77):
        super().__init__()
        self.model_name = "bert-base-uncased"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, text):
        output = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        hiden_state = self.model(output["input_ids"].to(self.device), output["attention_mask"].to(self.device))
        return hiden_state['pooler_output'][0]

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        return tokens


if __name__ == '__main__':
    model = BERTEmbedding()
    output = model('This is an example text.')
    print(output.shape)