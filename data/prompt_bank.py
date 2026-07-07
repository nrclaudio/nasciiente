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
    "toucan", "pelican", "stork", "crow", "raven", "dove", "hummingbird",
    "woodpecker", "ostrich", "turkey", "goose", "seagull", "puffin",
    "walrus", "seal", "otter", "beaver", "badger", "mole", "porcupine",
    "armadillo", "sloth", "chameleon", "gecko", "iguana", "salamander",
    "toad", "starfish", "stingray", "swordfish", "eel", "squid", "clam",
    "scorpion", "grasshopper", "cricket", "mosquito", "moth", "wasp",
    "caterpillar", "worm", "centipede", "buffalo", "bull", "donkey",
    "llama", "alpaca", "zebra", "rhinoceros", "hippopotamus", "cheetah",
    "leopard", "jaguar", "panther", "hyena", "meerkat", "lemur",
    "chimpanzee", "orangutan", "reindeer", "antelope", "boar", "ferret",
    "chipmunk", "opossum", "platypus", "narwhal", "orca", "manatee",
    "falcon", "hawk", "vulture", "condor", "kingfisher", "sparrow",
    "robin", "bluejay", "cardinal", "magpie", "heron", "crane bird",
    "albatross", "kiwi bird", "roadrunner", "quail", "pheasant",
    "salmon", "trout", "tuna", "catfish", "pufferfish", "anglerfish",
    "piranha", "flying fish", "manta ray", "hammerhead shark",
    "sea urchin", "sea turtle", "hermit crab", "barnacle", "krill",
    "tarantula", "praying mantis", "firefly", "cicada", "beetle",
    "stag beetle", "weevil", "tick", "flea", "silkworm", "earwig",
    "gazelle", "ibex", "yak", "bison", "warthog", "mongoose",
    "wolverine", "lynx", "bobcat", "ocelot", "serval", "capybara",
    "tapir", "okapi", "gnu", "dingo", "coyote", "jackal", "polar bear",
    "grizzly bear", "sun bear", "red panda", "snow leopard", "puma",
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
    "lantern", "torch", "quill", "typewriter", "wheelbarrow", "shovel",
    "rake", "broom", "wrench", "screwdriver", "pliers", "saw", "drill",
    "paintbrush", "palette", "easel", "harp", "saxophone", "flute",
    "banjo", "accordion", "harmonica", "microphone", "headphones",
    "joystick", "robot arm", "satellite dish", "light bulb", "battery",
    "plug", "switch", "thermometer", "syringe", "stethoscope", "bandage",
    "pill", "toothbrush", "comb", "mirror", "perfume bottle", "watch",
    "necklace", "bracelet", "backpack", "suitcase", "wallet", "purse",
    "glove", "mitten", "scarf", "bow tie", "necktie", "belt", "zipper",
    "button", "needle and thread", "safety pin", "paperclip", "stapler",
    "ruler", "protractor", "abacus", "calculator", "magnifying glass",
    "binoculars", "megaphone", "whistle", "yo-yo", "spinning top",
    "teddy bear", "rocking horse", "pinwheel", "slingshot", "boomerang",
    "dartboard", "bowling pin", "baseball bat", "tennis racket",
    "dumbbell", "barbell", "skis", "sled", "surfboard", "fishing rod",
    "picture frame", "birdcage", "birdhouse", "mailbox", "trash can",
    "watering can", "flower pot", "swing", "seesaw", "slide",
    "kettle", "frying pan", "saucepan", "ladle", "whisk", "grater",
    "rolling pin", "cutting board", "corkscrew", "can opener",
    "toaster", "blender", "mixer", "fork", "spoon", "knife",
    "chopsticks", "plate", "bowl", "pitcher", "goblet", "flask",
    "thermos", "lunchbox", "picnic basket", "cauldron pot",
    "soccer ball", "basketball", "football", "volleyball", "golf club",
    "hockey stick", "cricket bat", "ping pong paddle", "badminton racket",
    "boxing glove", "ice skate", "roller skate", "helmet", "trampoline",
    "pogo stick", "hula hoop", "frisbee", "kite string", "chessboard",
    "domino", "jigsaw puzzle piece", "rubik's cube", "marble",
    "jack-in-the-box", "kaleidoscope", "music box", "wind chime",
    "hammock", "parasol", "deck chair", "campfire grill", "compass rose",
    "hourglass timer", "pocket watch", "grandfather clock", "gramophone",
    "cassette tape", "vinyl record", "film reel", "clapperboard",
    "paint roller", "chisel", "anvil blacksmith", "bellows", "loom",
    "spinning wheel", "candlestick", "oil lamp", "chandelier",
    "feather quill", "wax seal", "treasure map", "spyglass", "sextant",
    "ship wheel", "life preserver", "diving helmet", "oxygen tank",
    "parachute", "jetpack", "telescope on tripod", "weather vane",
]

FOOD = [
    "apple", "banana", "cherry", "pear", "orange", "lemon", "grapes",
    "strawberry", "watermelon", "pineapple", "coconut", "peach", "plum",
    "carrot", "pumpkin", "corn", "pepper", "onion", "tomato", "broccoli",
    "eggplant", "pea pod", "bread loaf", "croissant", "pretzel",
    "baguette", "cheese wedge", "fried egg", "pancakes", "waffle",
    "cake", "cupcake", "donut", "cookie", "pie", "ice cream cone",
    "lollipop", "candy cane", "chocolate bar", "pizza slice",
    "hamburger", "hot dog", "taco", "sushi roll", "noodle bowl",
    "coffee cup", "milk bottle", "milkshake", "popcorn bucket",
    "honey pot", "salt shaker", "chef hat",
    "avocado", "mango", "kiwi fruit", "fig", "pomegranate", "apricot",
    "blueberry", "raspberry", "melon", "grapefruit", "lime", "papaya",
    "radish", "turnip", "beet", "cabbage", "cauliflower", "lettuce",
    "cucumber", "zucchini", "artichoke", "asparagus", "leek", "garlic",
    "ginger root", "chili pepper", "olive", "peanut", "walnut", "almond",
    "chestnut", "cashew", "sunflower seed", "rice bowl", "dumpling",
    "spring roll", "burrito", "quesadilla", "falafel", "kebab",
    "meatball", "drumstick", "fried chicken", "bacon strip", "omelette",
    "sandwich", "bagel", "muffin", "brownie", "macaron", "gingerbread man",
    "pudding", "jelly", "cheesecake", "tiramisu", "croquette", "paella",
    "teacup", "teapot with steam", "wine bottle", "beer mug", "cocktail",
    "juice box", "soda can", "water glass", "espresso cup",
]

PEOPLE = [
    "face", "smiley face", "eye", "hand", "footprint", "baby", "king",
    "queen", "jester", "pirate", "ninja", "astronaut", "chef", "farmer",
    "samurai", "viking", "cowboy", "ballerina", "juggler", "acrobat",
    "guitarist", "drummer", "fisherman", "archer", "boxer", "runner",
    "swimmer", "skier", "surfer", "snowman", "scarecrow", "mummy",
    "vampire", "clown", "detective", "graduate", "firefighter",
    "police officer", "sailor", "diver", "climber", "skater",
]

VEHICLES = [
    "car", "truck", "bus", "bicycle", "motorcycle", "train", "tram", "boat",
    "sailboat", "ship", "canoe", "submarine", "airplane", "helicopter",
    "hot air balloon", "rocket", "tractor", "tank", "skateboard", "scooter",
    "fire truck", "ambulance", "police car", "taxi", "van", "jeep",
    "race car", "limousine", "bulldozer", "crane truck", "forklift",
    "steam locomotive", "cable car", "rickshaw", "carriage", "chariot",
    "gondola", "ferry", "yacht", "raft", "kayak", "jet ski", "glider",
    "biplane", "fighter jet", "zeppelin", "space shuttle", "lunar rover",
    "snowmobile", "unicycle", "wheelchair", "stroller", "shopping cart",
]

NATURE = [
    "tree", "pine tree", "palm tree", "cactus", "flower", "rose", "tulip",
    "sunflower", "mushroom", "leaf", "acorn", "mountain", "volcano", "hill",
    "wave", "cloud", "sun", "moon", "crescent moon", "planet", "comet",
    "snowflake", "raindrop", "lightning bolt", "rainbow", "island",
    "feather", "seashell", "bone", "skull",
    "oak tree", "willow tree", "bonsai tree", "daisy", "orchid", "lily",
    "dandelion", "clover", "fern", "bamboo", "maple leaf", "pinecone",
    "tornado", "iceberg", "glacier", "waterfall", "river", "geyser",
    "sand dune", "coral", "spider web", "bird nest", "beehive", "cave",
    "crystal", "gemstone", "meteor", "constellation", "galaxy",
    "saturn", "eclipse", "sunrise", "campfire", "flame",
    "birch tree", "sequoia", "mangrove", "olive tree", "cherry blossom",
    "wheat stalk", "corn stalk", "vine", "ivy", "thistle", "reed",
    "lotus", "water lily", "poppy", "lavender", "hibiscus", "iris flower",
    "carnation", "peony", "marigold", "snowdrop", "crocus", "moss",
    "lichen", "toadstool", "truffle", "seaweed", "kelp", "driftwood",
    "boulder", "cliff", "canyon", "fjord", "atoll", "oasis", "marsh",
    "lagoon", "tide pool", "hot spring", "stalactite", "quicksand",
    "avalanche", "sandstorm", "monsoon cloud", "aurora", "milky way",
    "shooting star", "supernova", "black hole", "full moon", "half moon",
    "mars", "jupiter", "mercury planet", "venus planet", "neptune",
]

BUILDINGS = [
    "house", "cottage", "castle", "tower", "lighthouse", "windmill",
    "water mill", "barn", "church", "temple", "pagoda", "pyramid", "bridge",
    "arch", "fountain", "well", "tent", "igloo", "skyscraper", "fence",
    "gate", "staircase",
    "cabin", "hut", "palace", "fort", "watchtower", "clock tower",
    "bell tower", "mosque", "cathedral", "monastery", "obelisk",
    "colosseum", "amphitheater", "aqueduct", "dam", "silo", "greenhouse",
    "gazebo", "pavilion", "bandstand", "pier", "dock", "harbor",
    "ferris wheel", "carousel", "roller coaster", "circus tent",
    "phone booth", "bus stop", "street lamp", "traffic light",
    "fire hydrant", "manhole cover", "signpost", "sundial", "totem pole",
    "wind turbine", "oil rig", "space station", "observatory", "ruins",
]

FANTASY = [
    "dragon", "unicorn", "mermaid", "wizard", "witch", "knight", "robot",
    "alien", "ghost", "monster", "angel", "devil", "phoenix", "griffin",
    "sea serpent", "troll", "fairy", "genie lamp", "treasure chest",
    "magic wand", "crystal ball", "ufo",
    "pegasus", "centaur", "minotaur", "cyclops", "goblin", "gnome",
    "elf", "dwarf", "golem", "kraken", "hydra", "cerberus", "sphinx",
    "gargoyle", "werewolf", "zombie", "skeleton warrior", "grim reaper",
    "witch hat", "cauldron", "potion bottle", "spellbook", "rune stone",
    "excalibur", "holy grail", "flying carpet", "time machine",
]

SUBJECTS = (ANIMALS + OBJECTS + VEHICLES + NATURE + BUILDINGS + FANTASY
            + FOOD + PEOPLE)

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
    "a tiny {x}",
    "a giant {x}",
    "an old {x}",
    "a geometric {x}",
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


def _singles_pool():
    pool = {}
    for x in SUBJECTS:
        for template in SINGLE_TEMPLATES:
            pool[template.format(a=_article(x), x=x)] = None
    return list(pool)


def _counts_pool():
    return [f"{count} {_plural(x)}" for x in SUBJECTS
            for count in COUNT_WORDS]


def _pairs_pool():
    return [f"{_article(x)} and {_article(y)}"
            for x in SUBJECTS for y in SUBJECTS if x != y]


def _full_pool():
    """Every caption the bank can express (used by tests/statistics)."""
    return list(dict.fromkeys(_singles_pool() + _counts_pool()
                              + _pairs_pool()))


def _cycler(pool, rng):
    """Yield the pool endlessly, reshuffled each pass."""
    while True:
        chunk = pool[:]
        rng.shuffle(chunk)
        yield from chunk


def build_prompts(n, seed=0, mix=(0.75, 0.15, 0.10)):
    """Return n captions, deterministically for a given seed.

    mix = (singles, counts, pairs) sampling fractions. Singles dominate
    by design: few-step t2i models render one subject faithfully but tend
    to FUSE two-subject prompts into chimeras, so pair captions are kept
    as a small seasoning rather than the diet (composition is taught
    cleanly by the geometry stage's exact captions anyway). Each category
    cycles its own shuffled pool, so repeats pair the same caption with
    different generated images.
    """
    rng = random.Random(seed)
    cyclers = [_cycler(_singles_pool(), rng),
               _cycler(_counts_pool(), rng),
               _cycler(_pairs_pool(), rng)]
    prompts = []
    for _ in range(n):
        roll = rng.random()
        k = 0 if roll < mix[0] else (1 if roll < mix[0] + mix[1] else 2)
        prompts.append(next(cyclers[k]))
    return prompts
