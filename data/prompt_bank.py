"""Caption corpus for the synthetic data engine.

The trained model can only follow prompts it has seen the *shape* of, so
this bank aims for breadth over poetry: single drawable subjects, small
attribute variations, counts, and two-subject compositions. Captions stay
short and concrete — what a 48x80 glyph grid can actually express.

build_prompts(n, seed) returns n unique captions, deterministically.
"""

import random

ANIMALS = [
    "cat", "dog", "fish", "bird", "owl", "duck", "swan", "eagle", "penguin",
    "rabbit", "mouse", "elephant", "giraffe", "horse", "cow", "pig", "sheep",
    "goat", "lion", "tiger", "bear", "panda", "koala", "monkey", "gorilla",
    "fox", "wolf", "deer", "moose", "camel", "kangaroo", "crocodile",
    "turtle", "frog", "snake", "lizard", "dinosaur", "shark", "whale",
    "dolphin", "octopus", "crab", "lobster", "jellyfish", "seahorse",
    "butterfly", "bee", "ant", "spider", "snail", "dragonfly", "ladybug",
    "bat", "hedgehog", "squirrel", "raccoon", "skunk", "flamingo", "peacock",
    "parrot", "rooster", "hen", "chick", "rat", "hamster",
]

OBJECTS = [
    "cup", "mug", "teapot", "bottle", "wine glass", "vase", "lamp", "candle",
    "clock", "hourglass", "key", "lock", "book", "pencil", "scissors",
    "hammer", "axe", "sword", "shield", "bow and arrow", "umbrella", "hat",
    "top hat", "crown", "glasses", "shoe", "boot", "shirt", "dress", "sock",
    "chair", "table", "bench", "bed", "door", "window", "ladder", "bucket",
    "basket", "gift box", "balloon", "kite", "flag", "bell", "anchor",
    "wheel", "gear", "magnet", "telescope", "microscope", "camera", "phone",
    "computer", "television", "radio", "guitar", "violin", "piano", "drum",
    "trumpet", "chess piece", "dice", "playing card", "trophy", "medal",
    "envelope", "scroll", "map", "compass", "coin", "ring", "diamond",
    "heart", "star", "arrow", "question mark",
]

VEHICLES = [
    "car", "truck", "bus", "bicycle", "motorcycle", "train", "tram", "boat",
    "sailboat", "ship", "canoe", "submarine", "airplane", "helicopter",
    "hot air balloon", "rocket", "tractor", "tank", "skateboard", "scooter",
]

NATURE = [
    "tree", "pine tree", "palm tree", "cactus", "flower", "rose", "tulip",
    "sunflower", "mushroom", "leaf", "acorn", "mountain", "volcano", "hill",
    "wave", "cloud", "sun", "moon", "crescent moon", "planet", "comet",
    "snowflake", "raindrop", "lightning bolt", "rainbow", "island",
    "feather", "seashell", "bone", "skull",
]

BUILDINGS = [
    "house", "cottage", "castle", "tower", "lighthouse", "windmill",
    "water mill", "barn", "church", "temple", "pagoda", "pyramid", "bridge",
    "arch", "fountain", "well", "tent", "igloo", "skyscraper", "fence",
    "gate", "staircase",
]

FANTASY = [
    "dragon", "unicorn", "mermaid", "wizard", "witch", "knight", "robot",
    "alien", "ghost", "monster", "angel", "devil", "phoenix", "griffin",
    "sea serpent", "troll", "fairy", "genie lamp", "treasure chest",
    "magic wand", "crystal ball", "ufo",
]

SUBJECTS = ANIMALS + OBJECTS + VEHICLES + NATURE + BUILDINGS + FANTASY

# Attribute templates for a single subject. {a} is the article-ed subject.
SINGLE_TEMPLATES = [
    "{a}",
    "{a}",           # plain form weighted up — it's what users type most
    "a small {x}",
    "a big {x}",
    "{a} facing left",
    "{a} facing right",
    "a happy {x}",
    "a sleeping {x}",
    "a flying {x}",
    "a dancing {x}",
    "a simple {x}",
    "a cartoon {x}",
]

# Counts teach the composition/count control the class-name data never had
COUNT_WORDS = ["two", "three"]


def _article(word):
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


def _plural(word):
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _full_pool():
    """Every caption the bank can express, deduplicated, in a stable
    order: singles with attributes, counted subjects, ordered pairs.
    ~57k captions with the default subject lists."""
    pool = {}
    for x in SUBJECTS:
        for template in SINGLE_TEMPLATES:
            pool[template.format(a=_article(x), x=x)] = None
        for count in COUNT_WORDS:
            pool[f"{count} {_plural(x)}"] = None
    for x in SUBJECTS:
        for y in SUBJECTS:
            if x != y:
                pool[f"{_article(x)} and {_article(y)}"] = None
    return list(pool)


def build_prompts(n, seed=0):
    """Return n captions, deterministically for a given seed.

    The full pool is shuffled and cycled: for n up to the pool size
    (~57k) captions are unique; beyond that they repeat — which is what
    training wants anyway, since each repeat pairs the caption with a
    DIFFERENT generated image (many images per caption, like ImageNet's
    many images per class).
    """
    rng = random.Random(seed)
    pool = _full_pool()
    prompts = []
    while len(prompts) < n:
        chunk = pool[:]
        rng.shuffle(chunk)
        prompts.extend(chunk)
    return prompts[:n]
