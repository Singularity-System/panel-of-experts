import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
    # 150 TinyStories-style stories with varied vocabulary and structure
    stories = [
        # Animals (1-30)
        "Once upon a time, there was a little cat. The cat liked to play in the garden. It chased butterflies and climbed trees. The cat was very happy.",
        "There was a little dog named Spot. Spot loved to run in the park. He played fetch with his owner. Spot was the best dog ever.",
        "A rabbit hopped through the meadow. She found a patch of clover. The rabbit ate all the clover and felt full.",
        "There was a fish in a bowl. The fish was orange and gold. Every day, the fish swam in circles. The child liked to watch the fish.",
        "A bird built a nest in a tree. The bird collected twigs and grass. She laid three eggs in the nest. The eggs hatched into baby birds.",
        "There was a turtle in the pond. The turtle moved very slowly. A frog jumped on the turtle's shell. The turtle did not mind.",
        "A squirrel climbed a tall oak tree. He found a big acorn. The squirrel carried the acorn to his nest. He was ready for winter.",
        "There was a fox in the forest. The fox had red fur and sharp ears. He hunted for food every evening. The fox was very clever.",
        "A bear cub followed his mother. They walked through the woods. The mother bear showed the cub where to find honey. The cub ate the honey.",
        "There was a deer in the meadow. The deer had brown spots. She drank water from a stream. A butterfly landed near her.",
        "A penguin waddled on the ice. The ice was cold and white. The penguin slid on its belly. It was fun for the penguin.",
        "There was a monkey in the jungle. The monkey swung from vine to vine. He found a bunch of bananas. The monkey ate them all.",
        "A lion cub played in the grass. His mane was soft and fluffy. The cub roared but it sounded like a squeak. The mother lion smiled.",
        "There was an elephant by the river. The elephant sprayed water with its trunk. The water felt cool on the hot day. The elephant was happy.",
        "A giraffe reached the tall leaves. The leaves were at the top of the tree. The giraffe stretched its long neck. It ate the green leaves.",
        "There was a zebra on the plain. The zebra had black and white stripes. It ran very fast with the herd. The stripes helped it hide.",
        "A panda ate bamboo in the forest. The bamboo was fresh and green. The panda chewed slowly and loudly. It was a quiet day for the panda.",
        "There was a frog on a lily pad. The frog croaked in the pond. A dragonfly flew overhead. The frog jumped into the water.",
        "A snake slid through the grass. The grass was tall and green. The snake looked for a warm rock. It found one by the stream.",
        "There was a hedgehog in the bush. The hedgehog had prickly spines. It curled into a ball when scared. A child gently petted the hedgehog.",
        "A wolf howled at the moon. The moon was bright and round. The wolf's pack gathered around them. They howled together in the night.",
        "There was a mouse in the barn. The mouse found a piece of cheese. The mouse carried it to its hole. It was a feast for the mouse.",
        "A seal balanced a ball on its nose. The audience clapped and cheered. The seal clapped its flippers too. It loved the attention.",
        "There was an owl in the old tree. The owl had big yellow eyes. It hunted mice at night. The owl was wise and patient.",
        "A chick hatched from an egg. The eggshell cracked open slowly. The mother hen clucked softly. The chick was small and fluffy.",
        "There was a horse in the field. The horse had a brown coat. A girl brushed the horse's mane. The horse stood very still.",
        "A goat climbed the rocky hill. The hill was steep and rough. The goat found fresh grass at the top. It munched the grass happily.",
        "There was a pig in the mud. The pig loved to roll in the mud. It splashed water with its snout. The mud kept the pig cool.",
        "A sheep wandered from the flock. The sheep got lost in the fog. The sheepdog found the sheep and brought it back. The flock was safe again.",
        "There was a duck on the lake. The duck paddled in the water. Five ducklings followed behind. They all swam in a row.",
        # Kids & Daily Life (31-60)
        "Tim had a big red ball. He played with his ball in the park. Tim's friend Sam came to play too. They had fun throwing the ball back and forth.",
        "Mia found a box in the attic. Inside the box were old photos and a letter. The letter was from a long time ago. Mia loved reading the old words.",
        "Anna went to the beach. She built a big sandcastle. She put shells on top of the castle. The waves came and almost washed it away.",
        "Ben built a fort out of blankets and pillows. He crawled inside with his flashlight. It was cozy and warm. Ben read his favorite book in the fort.",
        "Tom opened a lemonade stand. He made lemonade from fresh lemons. His friends came to buy cups. Tom was happy to serve his friends.",
        "Lucy had a dream about flying. She flew over mountains and oceans. She saw clouds and stars. When she woke up, she wanted to fly again.",
        "Ella had a toy robot. The robot could walk and talk. Ella programmed it to dance. Everyone clapped when the robot danced.",
        "Jack planted a garden in spring. He put seeds in the dirt and watered them. Green sprouts came up in a week. Jack was so proud of his garden.",
        "Sophie baked cookies with her grandma. They mixed flour and sugar and chocolate chips. The cookies smelled delicious. Sophie ate three warm cookies.",
        "Noah went fishing with his dad. They sat by the lake all morning. Noah caught a big fish. His dad helped him reel it in.",
        "Emma rode her bike for the first time. Her dad held the seat and ran beside her. Emma pedaled faster and faster. She did it all by herself.",
        "Liam built a tower with his blocks. The tower got higher and higher. It fell down and Liam laughed. He built it again even taller.",
        "Ava visited the farm. She saw chickens and cows and pigs. Ava petted a baby goat. It was the best day ever.",
        "Mason went to the zoo. He saw a lion and a tiger and a bear. Mason fed the fish at the petting pond. He wanted to stay all day.",
        "Isabella found a four leaf clover. She showed it to her mom. Her mom said it was lucky. Isabella kept it in her pocket.",
        "Ethan drew a picture of his family. He used crayons and colored pencils. He gave the picture to his mom. She put it on the fridge.",
        "Charlotte made a card for her dad. She wrote I love you inside. She decorated it with stickers. Her dad framed the card.",
        "Aiden went sledding down the hill. The snow was deep and cold. Aiden went faster and faster. He laughed all the way down.",
        "Amelia had a birthday party. Her friends came and played games. They ate cake and opened presents. Amelia blew out all the candles.",
        "Harper went camping in the woods. She pitched a tent with her brother. They roasted marshmallows over the fire. Harper saw a shooting star.",
        "Logan went to the library. He picked out five books about space. Logan read about planets and rockets and stars. He wanted to be an astronaut.",
        "Evelyn had a sleepover. Her best friend came over with a sleeping bag. They told stories and ate popcorn. Evelyn fell asleep laughing.",
        "Daniel flew a kite in the wind. The kite went higher and higher. It danced in the blue sky. Daniel held the string very tight.",
        "Abigail went apple picking. She picked the biggest red apples. Abigail's mom made apple pie with them. The pie was delicious.",
        "Henry built a snowman in the yard. He used a carrot for the nose. Henry added buttons and a scarf. The snowman looked funny.",
        "Ella lost her favorite toy. She looked under the bed and in the closet. Her dog brought the toy to her. Ella hugged her dog.",
        "Sebastian learned to swim. His teacher held him in the water. Sebastian kicked his legs and moved forward. He swam across the pool.",
        "Grace found a penny on the sidewalk. She picked it up and looked at it. Grace put it in her piggy bank. She wanted to save it.",
        "Oliver had a talent show. He sang a song and played the guitar. Everyone clapped and cheered. Oliver was so happy.",
        "Chloe went to a birthday party. There were balloons and streamers. Chloe played pin the tail on the donkey. She won a prize.",
        # Nature & Seasons (61-90)
        "There was a little garden in the city. People grew flowers and vegetables. Children played among the flowers. The garden made everyone smile.",
        "A leaf fell from the tree. It floated down to the ground. It joined many other leaves. Together they made a colorful carpet.",
        "Spring came to the valley. Flowers bloomed in every color. Birds built nests in the trees. The stream flowed with melted snow.",
        "Summer brought hot days and long nights. Children played in the sprinkler. They ate ice cream and swam in the pool. Summer was the best season.",
        "Autumn painted the trees red and gold. The air turned crisp and cool. Kids raked leaves into big piles. They jumped into the piles and laughed.",
        "Winter covered the ground in white snow. Children put on their coats and boots. They made snow angels in the yard. Winter was cold but fun.",
        "Rain fell on the roof all night. The next morning there were puddles everywhere. A frog sat in one of the puddles. It croaked happily.",
        "The sun came up over the hills. Golden light filled the valley. Roosters crowed and cows mooed. A new day had begun on the farm.",
        "Stars twinkled in the night sky. A boy lay on the grass and counted them. He saw a shooting star and made a wish. The wish was for a puppy.",
        "A rainbow appeared after the storm. It had seven bright colors. A girl pointed at the rainbow and gasped. She had never seen one so close.",
        "Snowflakes drifted down from the clouds. They covered everything in white. A boy stuck out his tongue to catch them. They tasted like cold air.",
        "The wind howled through the trees. Branches swayed and leaves flew. A family sat by the fire inside. They felt warm and safe.",
        "A creek bubbled over smooth stones. The water was cold and clear. A girl dipped her toes in the creek. It felt refreshing on a hot day.",
        "Cherry blossoms covered the park. Pink petals fell like gentle rain. People walked under the trees and took pictures. The park was beautiful.",
        "Fog rolled into the harbor. Ships sounded their horns in the mist. A lighthouse beam cut through the fog. The boats found their way home.",
        "A waterfall crashed into the pool below. Mist rose from the water. A child stood on the rocks and felt the spray. It was powerful and loud.",
        "Desert sun beat down on the sand. A cactus stood tall and spiny. A lizard scurried into the shade. The desert was harsh but beautiful.",
        "Thunder rumbled in the distance. Lightning flashed across the sky. A dog hid under the bed. The storm passed and the sun came out.",
        "Dew drops covered the morning grass. Each drop held a tiny rainbow. A spider web glistened between two flowers. The world sparkled in the dawn.",
        "A meadow stretched as far as the eye could see. Wildflowers dotted the green grass. Bees buzzed from flower to flower. It was peaceful and wild.",
        "Mountains rose above the clouds. Their peaks were covered in snow. An eagle circled near the top. The mountains were ancient and strong.",
        "The ocean waves crashed on the shore. Sandpipers ran along the water's edge. A girl collected seashells in a bucket. Each shell was unique and pretty.",
        "Sunset painted the sky in orange and pink. Birds flew home to their nests. The world grew quiet and still. Night was about to begin.",
        "A brook wound through the forest. Fish swam in the shallow water. A heron stood perfectly still by the edge. It waited for the right moment.",
        "Mushrooms grew after the rain. They pushed up through the damp soil. Some were brown and some were red. The forest floor was alive.",
        "A tornado warning came on the radio. Families went to their basements. The wind howled and the trees bent. The storm passed and everyone was safe.",
        "Geese flew south in a V formation. They honked as they passed overhead. Autumn was ending and winter was coming. The geese would return in spring.",
        "Dandelions pushed through the cracks in the sidewalk. Their yellow flowers brightened the gray concrete. A child blew the seeds into the wind. They floated away.",
        "A glacier slowly moved down the valley. The ice was blue and ancient. Calves broke off and fell into the lake. The glacier was always changing.",
        "A volcano erupted on the island. Lava flowed down the sides into the sea. Steam rose from the hot rock. The earth was powerful and unpredictable.",
        # Magic & Fantasy (91-120)
        "Sam found a magic pencil. Whatever Sam drew became real. Sam drew a butterfly and it flew away. Sam drew a flower and it bloomed.",
        "There was a little rocket ship. The rocket ship could fly to the moon. Every night, the rocket flew to the moon and back. The astronauts loved the trip.",
        "A dragon lived in the cave. The dragon breathed fire but was friendly. He warmed the village in winter. The villagers loved the dragon.",
        "Lily opened a door that shouldn't exist. Behind it was a world of talking animals. They invited her to tea. Lily had the most fun afternoon ever.",
        "There was a magic carpet. It could fly anywhere in the world. A boy rode it over mountains and oceans. He saw things no one had ever seen.",
        "Emma found a wish in a bottle. She rubbed it and a genie appeared. The genie granted her one wish. Emma wished for happiness for everyone.",
        "A wizard lived in the tower at the end of town. He made potions and cast spells. One day he turned the river purple. Everyone thought it was beautiful.",
        "There was a secret door in the tree. A girl found it while hiking. Inside was a library of magic books. She read one story and learned to fly.",
        "Jack planted a magic bean. It grew into a giant beanstalk overnight. Jack climbed to the top and found a cloud kingdom. The clouds were soft and bouncy.",
        "A fairy appeared in a burst of light. She was no bigger than a thumb. The fairy granted three wishes. A boy wished for a bike and a book and a friend.",
        "There was a mirror that showed the future. A girl looked into it and saw herself grown up. She looked happy and kind. It made her smile.",
        "A wizard's cat could talk. It told the wizard what to cook for dinner. The cat said fish would be nice. The wizard made grilled fish.",
        "There was a box that never emptied. Whatever you put in came back out ten times. A girl put in a cookie and got ten cookies. She shared them with friends.",
        "A ghost lived in the old house. But the ghost was friendly and funny. It told jokes and played pranks. The children loved visiting the ghost.",
        "There was a wand that could change colors. A witch used it to paint the sunset. She painted the clouds pink and purple. Everyone marveled at the colors.",
        "A boy found a map in a bottle. It showed the way to hidden treasure. He followed the map through the forest. At the end was a chest full of books.",
        "There was a singing flower in the garden. It hummed a soft tune every morning. Bees came to listen to the music. The garden was always merry.",
        "A girl found a key under a rock. The key opened a tiny door in the wall. Inside was a miniature village. The people in the village waved at her.",
        "There was a tree that grew candy. Lollipops hung from the branches and chocolate coins grew at the base. Children visited the tree every day. It was their favorite place.",
        "A mermaid swam in the coral reef. She had scales of blue and green. She sang songs that echoed through the ocean. The fish stopped to listen.",
        "There was a magic hourglass. When you turned it over, time went backward. A boy turned it to relive his birthday. He had the best day twice.",
        "A troll lived under the bridge. He asked riddles before letting people cross. A girl answered all three riddles. The troll let her pass with a bow.",
        "There was a cloud that could shape-shift. It turned into a dragon and a ship and a castle. Children pointed and laughed as the shapes changed. The cloud was an artist.",
        "A girl found a pair of winged shoes. She put them on and floated into the air. She flew over the city and the countryside. It was the most amazing thing she ever did.",
        "There was a song that made flowers grow. A singer performed it in the garden. Seeds sprouted and blossomed as she sang. The garden became a wonderland.",
        "A boy built a time machine from cardboard. When he pressed the button, the room filled with light. He saw dinosaurs and spaceships. Then his mom called him for dinner.",
        "There was a painting that came alive at night. The animals in the painting walked out and explored. In the morning they went back to the canvas. No one knew their secret.",
        "A witch made a potion that made people fly. Her friends drank it and floated to the ceiling. They laughed and spun in the air. The potion wore off by dinner.",
        "There was a book that wrote itself. Every page appeared as you read the last. A girl read it from start to finish. It was the best story she ever read.",
        "A genie lived in a lamp in the attic. A boy found the lamp and rubbed it. The genie offered three wishes. The boy wished for endless cookies.",
        # Objects & Imagination (121-150)
        "There was a little train. The train went chug chug chug up the hill. It carried packages to the town. The town needed the packages very much.",
        "A fish swam in the pond. The pond was clear and cool. The fish saw ducks swimming above. The fish liked the pond very much.",
        "Lily planted a seed in her garden. She watered it every day. Soon a green sprout appeared. Lily was so excited to see it grow.",
        "A turtle crawled across the road. A little girl helped the turtle cross. She put it gently on the grass. The turtle crawled away slowly.",
        "There was a little garden in the city. People grew flowers and vegetables. Children played among the flowers. The garden made everyone smile.",
        "A bear cub wandered into the forest. He met a rabbit who showed him berries. The bear cub ate many berries and was very full. He went home happy.",
        "A leaf fell from the tree. It floated down to the ground. It joined many other leaves. Together they made a colorful carpet.",
        "There was a little rocket ship. The rocket ship could fly to the moon. Every night, the rocket flew to the moon and back. The astronauts loved the trip.",
        "A puppy chased its tail in circles. The tail was fast and the puppy was faster. Finally the puppy caught it. The tail wagged happily.",
        "There was a kite in the sky. The wind pushed it higher and higher. A child held the string very carefully. The kite danced in the blue.",
        "A frog sat on a big green leaf. It waited for a fly to land. The fly came and the frog jumped. The frog caught the fly with its tongue.",
        "There was a box of crayons. Each crayon had a different color. A child drew a picture of the sun. The yellow crayon was the favorite.",
        "A butterfly opened its wings slowly. The wings were painted blue and black. It fluttered from flower to flower. The butterfly sipped nectar from each one.",
        "There was a balloon at the party. It was bright red and shiny. A boy let it go and it floated up. The balloon climbed higher than the roof.",
        "A candle flickered on the cake. The room went dark as everyone sang. A girl blew out all the candles. The room lit up and everyone cheered.",
        "There was a puzzle on the table. A child put the pieces together one by one. The picture showed a rainbow. When the last piece fit, the child smiled.",
        "A clock ticked on the wall. Its hands moved slowly around the face. A boy watched it while waiting for school to end. The clock finally struck three.",
        "There was a paper airplane in the wind. It sailed across the room. A girl made another one and threw it. They raced each other across the yard.",
        "A block tower fell with a crash. A baby clapped her hands and laughed. Her older brother rebuilt the tower. This time he made it a fortress.",
        "There was a magnifying glass on the table. A boy looked at a leaf through it. The leaf had tiny lines and dots. It looked like a map of a strange land.",
        "A maraca shook at the party. Music played and people danced. A girl shook her maraca to the rhythm. The sound was loud and fun.",
        "There was a drum in the corner. A boy picked up the sticks and started to play. Boom boom went the drums. His dog howled along.",
        "A xylophone sat in the music room. Each bar was a different color and note. A girl tapped them with mallets. The music sounded like raindrops.",
        "There was a harmonica in a drawer. A boy found it and blew into it. A sad but sweet tune filled the room. His grandmother taught him a song.",
        "A puppet danced on a stage. Its strings moved as a child pulled them. The puppet bowed and waved. The audience clapped and cheered.",
        "There was a kaleidoscope on the shelf. A girl looked through it and turned the dial. Colors and shapes swirled inside. It was like magic.",
        "A paper boat sailed in a puddle. A child pushed it gently with a stick. The boat floated across the water. It was a grand voyage.",
        "There was a shadow on the wall. A boy moved his hands and made shapes. The shadow became a bird and then a dog. The wall was his theater.",
        "A sticker book sat on the desk. A girl peeled a star sticker and put it on her hand. Her hand sparkled in the light. She loved her stickers.",
        "There was a music box on the dresser. When opened, a tiny tune played. A ballerina spun inside the lid. A girl listened to the music and smiled.",
        "A chalk drawing covered the driveway. A child used every color to draw a city. Cars and buildings and trees filled the gray concrete. It was temporary art.",
        "There was a telescope in the backyard. A boy looked through it at the moon. He could see craters and mountains. The moon was closer than he thought.",
        "A yo-yo went up and down on a string. A boy practiced for hours until he could do tricks. The yo-yo slept at the bottom of the string. It was his favorite toy.",
        "There was a bucket and spade at the beach. A boy dug a hole deep into the sand. He found a shell and a pebble. The hole filled with seawater.",
        "A paper chain hung from the ceiling. Each link was a different color. A girl made it link by link. It stretched across the whole room.",
        "There was a magnifying glass in the garden. A child looked at an ant carrying a crumb. The ant was strong and determined. It carried its prize home.",
        "A rubber duck floated in the bathtub. It squeaked when squeezed. A child made waves and the duck bobbed up and down. Bath time was playtime.",
        "There was a pinwheel in the breeze. It spun round and round in the wind. A child ran with the pinwheel in hand. It spun faster the faster she ran.",
        "A jigsaw puzzle lay on the table. Two siblings worked on it together. They found the edge pieces first. Slowly the picture emerged. It showed a sunset.",
    ]
    return stories


def collate_fn(batch, pad_value=0):
    max_len = max(x.size(0) for x in batch)
    padded, masks = [], []
    for x in batch:
        pad_len = max_len - x.size(0)
        padded.append(torch.nn.functional.pad(x, (0, pad_len), value=pad_value))
        masks.append(torch.cat([torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    input_ids = torch.stack(padded)
    attention_mask = torch.stack(masks)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def make_data_loader(tokenizer, config, batch_size=8):
    texts = make_demo_data()
    ds = TextDataset(texts, tokenizer, config.max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
