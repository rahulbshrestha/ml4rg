import torch
import torch.nn as nn


class EncoderForClassification(torch.nn.Module):
    def init(self, pretrained_model, num_labels=2):
        super(EncoderForClassification, self).init()
        self.encoder = pretrained_model.bert
        self.linear = nn.Linear(pretrained_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        last_hidden_state = outputs.last_hidden_state  # The last hidden state of the encoder
        cls_output = last_hidden_state[:, 0, :]  # Take the [CLS] token representation
        logits = self.linear(cls_output)

        if output_hidden_states:
            return logits, outputs.hidden_states
        else:
            return logits


class EncoderForRegression(nn.Module):
    def __init__(self, pretrained_model):
        super(EncoderForRegression, self).__init__()
        self.encoder = pretrained_model.bert
        self.linear = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        last_hidden_state = outputs.last_hidden_state  # The last hidden state of the encoder
        pooled_output = last_hidden_state[:, 0]  # Take the [CLS] token representation
        regression_output = self.linear(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(regression_output.view(-1), labels.view(-1))

        if output_hidden_states:
            return (loss, regression_output, outputs.hidden_states) if loss is not None else (regression_output, outputs.hidden_states)
        else:
            return (loss, regression_output) if loss is not None else regression_output


class EncoderForRegression_old(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(EncoderForRegression_old, self).__init__()
        self.encoder = pretrained_model.bert
        self.linear = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        last_hidden_state = outputs.last_hidden_state  # The last hidden state of the encoder
        pooled_output = last_hidden_state[:, 0]  # Take the [CLS] token representation
        regression_output = self.linear(pooled_output)

        if output_hidden_states:
            return regression_output, outputs.hidden_states
        else:
            return regression_output


class HookedEncoderForRegression(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        # make a hook to extract intermediate model outputs
        hook_list = []
        self.embedding = [None]
        def hook_fn(model, input, output):
            self.embedding[0] = output
        hook_list.append(self.lm.bert.encoder.layer[10].register_forward_hook(hook_fn))
        self.linear = torch.nn.LazyLinear(1)

    def forward(self, x):
        self.lm(x)
        embed = (self.embedding[0][0]).mean(axis=1)
        return self.linear(embed).squeeze()