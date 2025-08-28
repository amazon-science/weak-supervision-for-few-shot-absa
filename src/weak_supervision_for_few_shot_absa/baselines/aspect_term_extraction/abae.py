from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
See https://aclanthology.org/P17-1036.pdf
"""


class ABAE(nn.Module):
    def __init__(self, number_of_aspects, dim=300) -> None:
        super().__init__()
        self.dim = dim
        self.pad_token = "[PAD]"
        self.K = number_of_aspects
        self.embeddings = nn.Embedding(10000, dim)
        self.M = nn.Parameter(torch.empty(size=(dim, dim)))
        self.T = nn.Parameter(torch.empty(size=(dim, self.K)))
        self.W = nn.Linear(in_features=dim, out_features=self.K)
        self.lmbd = 0.2

        nn.init.kaiming_uniform_(self.M.data)
        nn.init.kaiming_uniform_(self.T.data)

    def get_sentence_embedding(
        self, sentence_encoded: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:

        # (<batch_size>, <max_length>, 1)
        y_s = (
            sentence_encoded.sum(dim=1) / torch.count_nonzero(sentence_encoded, dim=1)
        ).unsqueeze(2)

        #
        # (<batch_size>, <max_length>, 1)
        d_i = torch.matmul(sentence_encoded, torch.matmul(self.M, y_s))
        # Mask the [PAD] tokens; No attention there
        d_i[~attention_mask] = -100

        # The attention weights. See (3) from the paper.
        # (<batch_size, <max_length>, 1)
        alpha_i = F.softmax(d_i, dim=1)

        # The sentence embedding
        # A weighted average over all the constituent words
        # (<batch_size>, <embedding_size>)
        z_s = (alpha_i * sentence_encoded).sum(dim=1)

        return z_s

    """
    We define the sentence embedding zs 
    """

    def forward(
        self,
        sentence_encoded: torch.tensor,
        attention_mask: torch.tensor,
        negative_embedding: torch.tensor,
    ):
        z_s = self.get_sentence_embedding(sentence_encoded, attention_mask)

        p_t = F.softmax(self.W(z_s), dim=1)
        r_s = torch.matmul(self.T, p_t.unsqueeze(dim=2)).squeeze(dim=2)

        return self.compute_loss(
            r_s,
            z_s,
            negative_embedding.sum(dim=1) / negative_embedding.count_nonzero(dim=1),
        )

    def compute_loss(
        self, reconstruction_embedding, sentence_embedding, negative_embedding
    ):
        pos = (sentence_embedding * reconstruction_embedding).sum(dim=1)
        neg = (negative_embedding * reconstruction_embedding).sum(dim=1)

        reconstruction_loss = F.relu(torch.sum(1 - pos + neg))
        orthogonality_regularizer = torch.norm(
            self.T @ self.T.T - torch.eye(self.T.shape[0])
        )

        return reconstruction_loss + self.lmbd * orthogonality_regularizer

    def pad_sequences(
        self, sentences: List[List[str]], pad_token="[PAD]"
    ) -> List[List[str]]:
        max_length = max(len(x) for x in sentences)
        padded_sentences = [s + [pad_token] * (max_length - len(s)) for s in sentences]
        return padded_sentences


sentence_encoded = torch.rand(8, 20, 300)
attention_mask = torch.ones(8, 20).bool()
attention_mask[3:, 15:] = False
attention_mask[2, 6:] = False
negative_embedding = torch.randn_like(sentence_encoded)
abae = ABAE(50, 300)
print(abae.forward(sentence_encoded, attention_mask, negative_embedding))
