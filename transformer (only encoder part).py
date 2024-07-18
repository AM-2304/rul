Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
... import math
... import copy
... import torch
... import torch.nn as nn
... 
... class Transformer(nn.Module):
... 
...     def __init__(self, model_dimension, src_vocab_size, number_of_heads, number_of_layers, dropout_probability, log_attention_weights=False):
...         super().__init__()
... 
...         self.src_embedding = Embedding(src_vocab_size, model_dimension)
...         self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
... 
...         mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
...         pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
...         encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
... 
...         self.encoder = Encoder(encoder_layer, number_of_layers)
...         self.init_params()
... 
...     def init_params(self, default_initialization=False):
...         if not default_initialization:
...             for name, p in self.named_parameters():
...                 if p.dim() > 1:
...                     nn.init.xavier_uniform_(p)
... 
...     def forward(self, src_token_ids_batch, src_mask):
...         src_representations_batch = self.encode(src_token_ids_batch, src_mask)
...         return src_representations_batch
... 
...     def encode(self, src_token_ids_batch, src_mask):
...         src_embeddings_batch = self.src_embedding(src_token_ids_batch)
...         src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)
...         src_representations_batch = self.encoder(src_embeddings_batch, src_mask)
...         return src_representations_batch
... 
... 
... class Encoder(nn.Module):
... 
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src_embeddings_batch, src_mask):
        src_representations_batch = src_embeddings_batch

        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, src_representations_batch, src_mask):
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch


class MultiHeadedAttention(nn.Module):
     def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

       
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        
        attention_weights = self.softmax(scores)

        attention_weights = self.attention_dropout(attention_weights)

        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

       
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        
        token_representations = self.out_projection_net(reshaped)

        return token_representations


#
# Input modules
#


class SublayerLogic(nn.Module):
     def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, representations_batch, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))



class PositionwiseFeedForwardNet(nn.Module):
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))



class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        embeddings = self.embeddings_table(token_ids_batch)

        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        
        return self.dropout(embeddings_batch + positional_encodings)


#
# Helper model functions
#


# def get_clones(module, num_of_deep_copies) {added below in the code}
    


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


# Testing the correctness of the transformer model - feel free to ignore - I used it during model development
if __name__ == "__main__":
    use_big_transformer = False

    # Dummy data
    src_vocab_size = 11
    trg_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=BIG_MODEL_DIMENSION if use_big_transformer else BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BIG_MODEL_NUMBER_OF_HEADS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BIG_MODEL_NUMBER_OF_LAYERS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BIG_MODEL_DROPOUT_PROB if use_big_transformer else BASELINE_MODEL_DROPOUT_PROB
    )

    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and trg were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer)}')

    out = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)

def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

def count_parameters(model):
     return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


"""# Testing the correctness of the transformer model - feel free to ignore - I used it during model development
if __name__ == "__main__":
    use_big_transformer = False

    # Dummy data
    src_vocab_size = 11
    trg_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=BIG_MODEL_DIMENSION if use_big_transformer else BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BIG_MODEL_NUMBER_OF_HEADS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BIG_MODEL_NUMBER_OF_LAYERS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BIG_MODEL_DROPOUT_PROB if use_big_transformer else BASELINE_MODEL_DROPOUT_PROB
    )

"""
    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and trg were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer)}')

