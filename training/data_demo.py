import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, return_tensors="pt", max_length=self.max_seq_len, truncation=True)
        return encoded["input_ids"].squeeze(0)


def make_demo_data():
    stories = [
        "Once upon a time, there was a little cat. The cat liked to play in the garden. It chased butterflies and climbed trees. The cat was very happy.",
        "Tim had a big red ball. He played with his ball in the park. Tim's friend Sam came to play too. They had fun throwing the ball back and forth.",
        "There was a little bird. The bird could sing a beautiful song. Every morning, the bird sang in the tree. All the animals loved to listen.",
        "Mia found a box in the attic. Inside the box were old photos and a letter. The letter was from a long time ago. Mia loved reading the old words.",
        "A dog named Max lived on a farm. Max loved to run in the fields. He chased the sheep and herded them back to the pen. Max was a good dog.",
        "Lily planted a seed in her garden. She watered it every day. Soon a green sprout appeared. Lily was so excited to see it grow.",
        "There was a little train. The train went chug chug chug up the hill. It carried packages to the town. The town needed the packages very much.",
        "Ben built a fort out of blankets and pillows. He crawled inside with his flashlight. It was cozy and warm. Ben read his favorite book in the fort.",
        "A fish swam in the pond. The pond was clear and cool. The fish saw ducks swimming above. The fish liked the pond very much.",
        "Anna went to the beach. She built a big sandcastle. She put shells on top of the castle. The waves came and almost washed it away.",
        "There was a little rocket ship. The rocket ship could fly to the moon. Every night, the rocket flew to the moon and back. The astronauts loved the trip.",
        "Sam found a magic pencil. Whatever Sam drew became real. Sam drew a butterfly and it flew away. Sam drew a flower and it bloomed.",
        "A bear cub wandered into the forest. He met a rabbit who showed him berries. The bear cub ate many berries and was very full. He went home happy.",
        "Ella had a toy robot. The robot could walk and talk. Ella programmed it to dance. Everyone clapped when the robot danced.",
        "A leaf fell from the tree. It floated down to the ground. It joined many other leaves. Together they made a colorful carpet.",
        "Tom opened a lemonade stand. He made lemonade from fresh lemons. His friends came to buy cups. Tom was happy to serve his friends.",
        "A turtle crawled across the road. A little girl helped the turtle cross. She put it gently on the grass. The turtle crawled away slowly.",
        "Lucy had a dream about flying. She flew over mountains and oceans. She saw clouds and stars. When she woke up, she wanted to fly again.",
        "There was a little garden in the city. People grew flowers and vegetables. Children played among the flowers. The garden made everyone smile.",
        "A dragon lived in the cave. The dragon breathed fire but was friendly. He warmed the village in winter. The villagers loved the dragon.",
    ]
    return stories


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded = []
    masks = []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded_x = torch.nn.functional.pad(x, (0, pad_len), value=pad_value)
        mask = torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        padded.append(padded_x)
        masks.append(mask)
    input_ids = torch.stack(padded)
    attention_mask = torch.stack(masks)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def make_data_loader(tokenizer, config, batch_size=8):
    from torch.utils.data import DataLoader

    texts = make_demo_data()
    ds = TextDataset(texts, tokenizer, config.max_seq_len)

    def collate(batch):
        return collate_fn(batch)

    return DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)
