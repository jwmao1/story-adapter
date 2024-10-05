import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from ip_adapter import StoryAdapterXL
import os
import random
import argparse


story1 = [
"a little white rabbit wearing green suit and two little white rabbits in a room.",
"a little green turtle and eggs on a beach.",
"a little white rabbit wearing green suit jumping in a room.",
"a little green turtle walking to the ocean from beach.",
"a little white rabbit wearing green suit running in the forest.",
"a little green turtle slowly running on the beach.",
"a little white rabbit wearing green suit running in marshes.",
"a little green turtle and sharks in the ocean.",
"a little green turtle running between rocks.",
"a little white rabbit wearing green suit running in the room.",
"a little green turtle climbing the hill near the lake.",
"a little white rabbit wearing green suit running in stormy night.",
"a little green turtle swimming with fish in ocean.",
"a little white rabbit wearing green suit jumping in the farm.",
"a little green turtle running in the mud.",
"a little white rabbit wearing green suit running with a sheep on the farm.",
"a little green turtle and whales in the ocean.",
"a little white rabbit wearing green suit jumping on tree stumps.",
"a little green turtle digging holes in the sand.",
"a little white rabbit wearing green suit and three little white rabbits running in village competition.",
"a little green turtle running with three little green turtles in railroad track.",
"a little white rabbit wearing green suit lifting dumbbells in room.",
"a little green turtle dancing at the gym.",
"a little white rabbit wearing green suit running on a mountain.",
"a little white rabbit wearing green suit running fast in snowy mountain.",
"a little green turtle swimming quickly in seafloor.",
"a little white rabbit wearing green suit running passes a duck in snowy mountain.",
"a little green turtle swimming passes a cuttlefish in seafloor.",
"a little white rabbit wearing green suit jumping over the cliff on the snowy mountain.",
"a little white rabbit wearing green suit running in a cave in an snowy mountain.",
"a little white rabbit wearing green suit hold a torch in a cave in an snowy mountain.",
"a little green turtle swimming past an undersea shipwreck.",
"a little green turtle running past a red flag in seafloor.",
"a little green turtle holding up a trophy made of fish bones in seafloor.",
"a little white rabbit wearing green suit picking flowers on an snowy mountain.",
"a little white rabbit wearing green suit running past a red flag in snowy mountain.",
"a little white rabbit wearing green suit holding up a trophy in the snowy mountain.",
"a little green turtle running on desert.",
"a little white rabbit wearing green suit buying shoes at the village store.",
"a little white rabbit wearing green suit and two black rabbits eating cake in room.",
"a little green turtle running passes a zebra in desert.",
"a little green turtle running across a small lake in the desert.",
"a little white rabbit wearing green suit running in the forest in the evening.",
"a little white rabbit wearing green suit jumping in the forest in the evening.",
"a little green turtle drinking water in the small lake in desert.",
"a little green turtle running passes a camel in desert.",
"a little white rabbit wearing green suit playing basketball in the forest in the evening.",
"a little white rabbit wearing green suit cycling in the forest in the evening.",
"a cheetah running past a red flag in the desert.",
"a cheetah holding up a trophy on the podium in the desert.",
"a little green turtle running past a red flag in the desert.",
"close shot of a little green turtle crying in the desert.",
"a little white rabbit wearing green suit watching film in the room evening.",
"a little white rabbit wearing green suit playing computer game in the room evening.",
"a little white rabbit wearing green suit reading comic book in the room evening.",
"a little green turtle running in the beach in the stormy night.",
"a little green turtle doing yoga in the beach in stormy night.",
"a little green turtle running on a treadmill in the beach in stormy night.",
"a little green turtle doing dumbbell training on the beach in the rainy day.",
"a little white rabbit wearing green suit sleeping on the bed in the room in the rainy day.",
"a large number of audience came to the village tournament.",
"a little white rabbit wearing green suit and a little green turtle prepare at the starting line in a village.",
"a referee announces the race rules in a village.",
"a little white rabbit wearing green suit running cross the white starting line in a village.",
"a little green turtle slowly takes its first step in a village.",
"a little white rabbit wearing green suit running passes a little green turtle in a village road.",
"a little green turtle running moves forward in a village road.",
"a little white rabbit wearing green suit running passes a pig in a farm.",
"a little green turtle keeps moving steadily in a farm.",
"a little white rabbit wearing green suit running in the forest.",
"a little white rabbit wearing green suit running passes a eagle in the forest.",
"a little white rabbit wearing green suit running passes a horse in the forest.",
"a little white rabbit wearing green suit walking in the forest.",
"a little green turtle running in the forest.",
"a little white rabbit wearing green suit sitting by a tree in the forest.",
"a little white rabbit wearing green suit sleeping near a tree in the forest.",
"a little green turtle running passes a lion in the forest.",
"a little green turtle running passes a cheetah in the forest.",
"a little green turtle running passes a tiger in the forest.",
"a little green turtle running on a railroad.",
"a little white rabbit wearing green suit eating cake in room in dream.",
"a little white rabbit wearing green suit jumping in the forest.",
"a little green turtle running on a hill.",
"a little green turtle running passes a chicken on a hill.",
"a little green turtle swimming in a lake.",
"a little white rabbit wearing green suit running on a railroad.",
"a little green turtle swimming passes a frog in a lake.",
"a little white rabbit wearing green suit running passes a deer on a railroad.",
"a little green turtle swimming near coral in a lake.",
"a little white rabbit wearing green suit running passes a cheetah on a hill.",
"a little white rabbit wearing green suit running passes a snake on a hill.",
"a little green turtle near the red flag in mountain path.",
"a little green turtle running past a red flag in mountain path.",
"a little white rabbit wearing green suit swimming in a lake.",
"a little white rabbit wearing green suit running in mountain path.",
"a little white rabbit wearing green suit running past a red flag in mountain path.",
"close shot of a little white rabbit wearing green suit crying in mountain path.",
"many people cheered on the mountain path.",
"a little green turtle holding up a trophy on the podium in the village.",
"close shot of a little little green turtle smile in village.",
]

story2 = [
"One Winnie the Pooh wake up in the bed",
"One Winnie the Pooh brushing teeth in the room",
"One Winnie the Pooh Bathing in the bathtub in the room.",
"One Winnie the Pooh eating a breakfast in the room.",
"One Winnie the Pooh walks to the door",
"One Winnie the Pooh opens the door",
"One Winnie the Pooh walks out of the house",
"One Winnie the Pooh walks down the path",
"One Winnie the Pooh sees a white Rabbit",
"One Winnie the Pooh waves in the forest",
"One Winnie the Pooh enters the woods",
"One Winnie the Pooh running in the forest",
"One Winnie the Pooh continues walking",
"Piglet wearing red jeans angling in a lake",
"Piglet wearing red jeans waves in the forest",
"Winnie the Pooh hugs Piglet wearing red jeans",
"Close shot of a Piglet wearing red jeans smiles in the forest",
"Winnie the Pooh and Piglet wearing red jeans walking in the forest",
"One Winnie the Pooh sees a big tree",
"A beehive hanging from a big tree.",
"Winnie the Pooh walks to the big tree",
"Winnie the Pooh climbs the big tree",
"One Winnie the Pooh holding a beehive near a big tree",
"One Winnie the Pooh eating honey in a beehive",
"One Winnie the Pooh sees bees near a big tree",
"bees attacking Winnie the Pooh near a big tree",
"Winnie the Pooh reaches out to grab a bee on a big tree",
"The bee flies away",
"Winnie the Pooh climbs down the tree",
"Winnie the Pooh and Piglet wearing red jeans walking in a small river",
"Piglet waves near a small river",
"One Winnie the Pooh builds a small raft",
"a frog on the small river",
"One Winnie the Pooh float on the river",
"One Winnie the Pooh paddle quickly on the raft",
"One Winnie the Pooh sees a waterfall in the distance",
"One Winnie the Pooh walking in the waterfall",
"One Winnie the Pooh sees a small hill",
"One Winnie the Pooh walking on the hill",
"One Winnie the Pooh sitting by a tree on the hill",
"One Winnie the Pooh running on the hill",
"a mysterious cave in a hill",
"One Winnie the Pooh walking on the hill in a stormy night",
"One Winnie the Pooh enters the cave in a stormy night",
"One Winnie the Pooh starting a fire with a flint in the cave in a stormy night.",
"heavily rain in the forest in a stormy night.",
"One Winnie the Pooh holding a torch walking in the cave in a stormy night.",
"A green book in a table in the cave in a stormy night.",
"One Winnie the Pooh reading a green book in the cave in a stormy night.",
"One Winnie the Pooh looking a map in the cave in a stormy night.",
"Winnie the Pooh sleeping in the cave in a stormy night.",
"sun rises in the forest.",
"One Winnie the Pooh eatting a bread in the cave.",
"Winnie the Pooh holding a green book on the hill.",
"One Winnie the Pooh walking in the forest.",
"a snake in the forest.",
"One Winnie the Pooh jumped up in fear in the forest.",
"One Winnie the Pooh running in the forest in fear.",
"Dense fog appears in the forest.",
"One Winnie the Pooh takes out a compass",
"One Winnie the Pooh walking in a patch of tall grass.",
"One Winnie the Pooh uses a machete clearing the grass",
"One Winnie the Pooh floatting on a small raft in a small river",
"a crocodile swimming in a little river",
"One Winnie the Pooh and one crocodile floatting in a river",
"One Winnie the Pooh walking in the marsh.",
"a crocodile sinking into a marsh",
"One Winnie the Pooh walking on a bridge.",
"One Winnie the Pooh sees a tower in the distance",
"One Winnie the Pooh walk in the base of the tower",
"A red book placed in the ground near the base of the tower",
"One Winnie the Pooh reading a red book in the base of the tower",
"One Winnie the Pooh looking a map in the base of the tower.",
"One Winnie the Pooh walking in a jail cell",
"One Winnie the Pooh touch a door in a jail cell",
"a mysterious cave in a jail cell",
"One Winnie the Pooh enter a cave in a jail cell",
"One Winnie the Pooh walking in the sewer",
"One Winnie the Pooh holding a torch in the sewer",
"mouse running in the sewer.",
"One Winnie the Pooh fell down in the sewer",
"One Winnie the Pooh walked out of the sewer",
"One Winnie the Pooh sees the sunset",
"One Winnie the Pooh sets up a camp in stormy night",
"One Winnie the Pooh lights a campfire in stormy night",
"One Winnie the Pooh reading a red book by the fire in the forest in stormy night",
"One Winnie the Pooh fall asleep by the fire in stormy night",
"Winnie the Pooh flying in a hot-air balloon.",
"A hot air balloon flying in the beautiful view of a cloud and a rainbow.",
"hot-air balloon in a canyon.",
"One Winnie the Pooh walking in a canyon.",
"Piglet wearing red jeans give a blue book to Winnie the Pooh in a canyon",
"Piglet wearing red jeans waving in a canyon.",
"One Winnie the Pooh reading a blue book in a canyon.",
"One Winnie the Pooh walked by the river near the canyon.",
"Close shot of treasure chest near a lake, cayon.",
"One Winnie the Pooh and a treasure chest near a lake, cayon",
"One Winnie the Pooh opens the treasure chest near a lake, cayon",
"Close shot of Gold, silver and jewelry are in the treasure chest near a lake, cayon.",
"One Winnie the Pooh happly walking to a wood house in the forest.",
]

story3 = [
"Stormy night at sea, with raging winds and waves tossing the ship.",
"Stormy night at sea, Robinson swimming in the sea.",
"Stormy night at sea, Robinson swept away by waves, clutching a broken plank.",
"At dawn, Robinson lie on a deserted island beach.",
"Close-up of Robinson's face as he wakes on the island.",
"Robinson walk in the forest.",
"Robinson finds a stream in the forest.",
"Robinson holds the water of the stream in hand.",
"Robinson holds branches in the forest.",
"Robinson builds a wood shelter in the forest.",
"Robinson starting a fire with a flint in the forest.",
"Robinson sits by the fire in the forest.",
"Robinson picks fruit trees in the forest.",
"Robinson clearing a small plot of land in the forest.",
"Robinson watered the wheat in the forest.",
"Robinson catching rabbit in the forest.",
"Robinson tames goats in the forest.",
"goats eat grass in the forest.",
"Robinson roasting in the forest at night.",
"Robinson writing in the forest shelter at night.",
"Robinson uses hammers to repair a wood boat on the island.",
"Robinson attempts to sail out with wood boat.",
"wood boat forced back by storms and currents.",
"Robinson stood at the highest point in the forested mesas.",
"Big footprints appeared in the forest road.",
"Orangutans roasting in the forest",
"Friday was tied up with a rope in the forest.",
"Robinson and Friday run in the forest.",
"orangutans run in the forest.",
"Robinson and Friday talk on the beach.",
"Robinson and Friday stone shelters in the forest.",
"Robinson making spear in the island.",
"orangutans discover camp in the forest.",
"orangutans fire camp in the forest.",
"Robinson and orangutans in the forest.",
"Robinson punches orangutans in the forest.",
"Friday find European ship approaching deserted beaches.",
"Crew lands on the island.",
"Crew shoot orangutans in the forest.",
"Friday waving to Robinson on the beach.",
"Robinson waving on the ship.",
"Robinson walk in the city streets.",
"Rubenson talk with the journalist.",
"Robinson hug with his wife in the city street.",
"Robinson writing diary at home.",
"Robinson see newspaper at home.",
"Robinson eat dinner at home.",
"Robinson walks in the garden.",
"Robinson sit in a church.",
"Robinson is sleeping in the bed in the house."
    ]

story4 = [
"One baby lying in a cradle on the deck on a ship.",
"On a luxury cruise ship, passengers are enjoying the sunshine on the deck on a ship.",
"One crew member holding a baby on deck on a ship.",
"One captain holding a baby on the cabin on a ship.",
"One little boy wearing black suit, holding a muppet on the cabin on a ship.",
"One little boy wearing black suit walking on the cabin on a ship.",
"Children playing on a luxury cruise ship deck.",
"One little boy wearing black suit, on a luxury cruise ship deck, looking dolphin in the sea.",
"One little boy wearing black suit, running on a luxury cruise ship deck.",
"One little boy wearing black suit, walking in a luxury cruise ship hall.",
"peoples dancing in the luxury cruise ship hall.",
"peoples look One man playing a piano in the luxury cruise ship hall.",
"One little boy wearing black suit, holding a newspaper in a luxury cruise ship hall.",
"One little boy wearing black suit, playing a piano in Unoccupied halls in a luxury cruise ship.",
"peoples look One man wearing black suit, playing a piano in a luxury cruise ship hall.",
"Many people are clapping and applauding in a luxury cruise ship hall.",
"One black man wearing black suit, walking in a harbor.",
"One black man wearing black suit, playing a piano in a luxury cruise ship hall.",
"One man wearing black suit playing a piano, and One black man wearing black suit playing a piano, in a luxury cruise ship hall.",
"One black man wearing black suit, sweating while playing the piano, in a luxury cruise ship hall.",
"One black man wearing black suit, perspire near a piano in luxury cruise ship hall.",
"One man wearing black suit, shaking hand with One black man wearing black suit, in a luxury cruise ship hall.",
"One man wearing black suit, looking the window in the cabin on a luxury cruise ship.",
"One man wearing black suit, walking on a deck of crowded luxury cruise ship.",
"peoples look One man wearing black suit, dancing on a luxury cruise ship hall.",
"One man wearing black suit, dancing with a woman wearing red dress, on a luxury cruise ship hall.",
"One man wearing black suit, and a woman wearing red dress, talking on a luxury cruise ship hall.",
"One man wearing black suit, and a woman wearing red dress, eating dinner on a luxury cruise ship dinning hall.",
"One man wearing black suit, singing to a woman wearing red dress, on a luxury cruise ship dinning deck.",
"One man wearing black suit, waving on a luxury cruise ship dinning deck, sadly.",
"One woman wearing red dress, disembarked a luxury cruise ship at the harbor, sadly.",
"One woman wearing red dress, carrying suitcases at the harbor, sadly.",
"One woman wearing red dress, waving at the harbor, sadly.",
"One woman wearing red dress, crying in the harbor.",
"One man wearing black suit, smoking in the cabin on a luxury cruise ship, sadly.",
"One man wearing black suit, drinking beer in the cabin on a luxury cruise ship, sadly.",
"One man wearing black suit, drinking beer at the bar in the cabin on a luxury cruise ship, sadly.",
"One man wearing black suit, disembarking a luxury cruise ship at the harbor.",
"One man wearing black suit, standing on a harbor near a luxury cruise ship.",
"One man wearing black suit, running on a luxury cruise ship deck.",
"One man wearing black suit, playing piano on a luxury cruise ship deck.",
"Many people are clapping and applauding on a luxury cruise ship hall.",
"One emperor and One man wearing black suit, playing the piano in a luxury cruise ship hall.",
"A Nobody's rusty luxury cruise ship at sea in a ship repair yard.",
"An construction worker wearning helmet and work clothes, placing bombshell on a Nobody's rusty luxury cruise ship deck.",
"An construction worker wearning helmet and work clothes, shouting on a Nobody's rusty luxury cruise ship deck.",
"One man wearing black suit, playing a rusty piano on a Nobody's rusty luxury cruise ship deck.",
"One Nobody's rusty luxury cruise ship exploded on the sea.",
"Sea slowly flooded One man wearing black suit in ballroom, in a Nobody's rusty luxury cruise ship.",
"One man wearing black suit, closed eyes and smiled in the seafloor."
]

story5 = [
    "In the colorful tent, the clown rolls and does a backflip.",
    "In the shining center stage, the acrobat flips in the air.",
    "In the elephant performance area, the trainer directs the elephant to lift a ball with its trunk.",
    "Beside the lion's cage, the trainer gently strokes the lion's mane.",
    "Under the brilliant lights, the magician waves his wand and conjures a white dove.",
    "In the lively audience seats, the children cheer excitedly.",
    "Next to the ring of fire, the trainer directs the tiger to jump through the hoop.",
    "In the circus backstage, the makeup artist applies bright makeup to the performers.",
    "Under the spotlight on stage, the clown dances and performs funny acts.",
    "In the front row of the audience, the little girl watches intently with her cotton candy.",
    "High in the air, the trapeze artist swings back and forth.",
    "In the center of the performance area, the horse trainer directs the horses to run around the ring.",
    "At the circus entrance, the clown hands out balloons to the children.",
    "In the dressing room backstage, the performers busily change costumes.",
    "In the center of the stage, the fire eater breathes out a plume of flame.",
    "Above the audience seats, a flock of white doves flies by.",
    "Outside the circus tent, colorful flags flutter in the wind.",
    "Under the dazzling spotlight, the dancers twirl gracefully.",
    "High on the saddle, the equestrian performer performs daring stunts.",
    "In front of the makeup mirror backstage, the clown carefully paints on a big smile.",
    "In the back row of the audience, parents happily watch the performance with their children.",
    "On the stage under the lights, the trainer directs the dog to jump over obstacles.",
    "High in the air on the stage, the aerialists catch each other mid-flight.",
    "Above the audience seats, colorful balloons slowly rise.",
    "In the center of the performance area, the elephant spins a hula hoop with its trunk.",
    "In the middle of the audience, a little boy claps excitedly.",
    "Under the stage lights, the fire breather showcases the art of flame.",
    "In the center of the stage, the acrobats form a human pyramid.",
    "In the front row of the audience, the little girl tightly hugs her doll.",
    "In the corner of the performance area, the clown comically slips and falls.",
    "In the dressing room backstage, the performers quickly change costumes.",
    "High in the air on the stage, the trapeze artist soars.",
    "In the center of the performance area, the magician pulls a scarf out of thin air.",
    "In the front row of the audience, the children cheer excitedly.",
    "Under the stage lights, the trainer directs the lion to jump through the hoop.",
    "In the back row of the audience, parents smile as they watch the show with their children.",
    "In the center of the performance area, the elephant blows a trumpet with its trunk.",
    "In the middle of the audience, a little boy joyfully waves his balloon.",
    "Under the stage lights, the acrobats spin in the air.",
    "In the center of the stage, the clown performs a funny pratfall.",
    "In the front row of the audience, the little girl watches the performance with anticipation.",
    "In the corner of the performance area, the horse trainer directs the horses to leap.",
    "On the stage with flashing lights, the fire breather showcases flame tricks.",
    "In the dressing room backstage, the performers hurriedly organize their outfits.",
    "High in the air on the stage, the aerialists catch each other in flight.",
    "Above the audience seats, colorful confetti rains down.",
    "In the center of the performance area, the magician pulls out doves from his hat.",
    "Under the stage lights, the trainer directs the dogs to perform tricks.",
    "High in the air on the stage, the acrobats perform flips.",
    "In the back row of the audience, parents happily watch the performance with their children."
]


story6 = [
    'A black man in a black jacket walks out of a closed house and observes the empty New York streets.',
    'A black Armored SUV car traveling in an empty abandoned city.',
    'A herd of deer running in the empty streets of an abandoned city.',
    'An black armored SUV car chases deer on the empty streets of an abandoned city.',
    'A black man in a black jacket shooting in the empty rode in an abandoned city.',
    'A black German Shepherd is jumping out of an black armored SUV on the road in a abandoned city.',
    'A black German Shepherd near a deer corpse on the empty road in a abandoned city.',
    'A black man in a black jacket looking at phone under the sunset in a empty abandon city.',
    'A black man in a black jacket light bonfires in an empty abandoned office.',
    'A black German Shepherd lying near a bonfire in an empty abandoned office.',
    'A zombie walk in a empty meeting room.',
    'Clos shot of a scared expression of A black man in a black jacket in an empty abandoned office.',
    'A black man in a black jacket walking in a empty abandoned hospital.',
    'A black man in a black jacket holding a drug in a empty abandoned hospital.',
    'A black man in a black jacket steps on broken glass in a empty abandoned hospital.',
    'A black man in a black jacket and A black German Shepherd running in the empty abandoned hospital.',
    'Three zombies running in empty abandoned hospital.',
    'A black man in a black jacket holding a drug crouching in a empty abandoned repository.',
    'A black man in a black jacket close the door in a empty abandoned repository.',
    'A black Armored SUV car near a empty abandoned hospital in evening.',
    'A black Armored SUV car in a mountain road in evening.',
    'A black man in a black jacket and one black German Shepherd in a empty abandoned manor in evening.',
    'A black man in a black jacket playing computer in a lab.',
    'A black man in a black jacket wake up in a bed in a abandoned room.',
    'A black man in a black jacket eating breakfast on a table in a abandoned room.',
    'A black German Shepherd eating bitterness on the ground in a abandoned room.',
    'A black man in a black jacket walking in a empty abandoned supermarket.',
    'A black man in a black jacket bend down picking up a snack in empty abandoned supermarket.',
    'A black German Shepherd running to a human corpse in empty abandoned supermarket.',
    'A black man in a black jacket picking a flashlight on shelve in empty abandoned supermarket.',
    'A zombie running in empty abandoned supermarket.',
    'A black man in a black jacket and A black German Shepherd running in empty abandoned supermarket.',
    'A black man in a black jacket blocking the door with a sofa outside a empty abandoned supermarket.',
    'A black man in a black jacket fixing car on the empty road in a abandoned city.',
    'A black man in a black jacket using a flashlight in a forest in evening.',
    'A black man in a black jacket barbeque in a forest in evening.',
    'A black German Shepherd eating a roast beef in a forest in evening.',
    'A black man in a black jacket using a microscope in the empty abandoned lab.',
    'A black man in a black jacket and A black German Shepherd in a empty abandoned park.',
    'A black man in a black jacket carrying a shotguns in a empty abandoned condos.',
    'A black man in a black jacket holding a shotguns checking the solar panel in a empty abandoned room.',
    'A black German Shepherd sitting on the steps in a empty abandoned condos.',
    'A black man in a black jacket putting a solar panel on the rooftop.',
    'A black German Shepherd biting a old toy in a empty abandoned room.',
    'A black man in a black jacket eating a canned on sofa in a empty abandoned room.',
    'many zombies walking on a empty abandoned road.',
    'A black man in a black jacket looking a zombie by window in a empty abandoned room.',
    'A black man in a black jacket holding a list near a computer in the empty abandoned lab.',
    'A black man in a black jacket walking in a empty abandoned library.',
    'A black man in a black jacket reading a book in a empty abandoned library.',
    'A black man in a black jacket writing a diary in a empty abandoned library.',
    'A black man in a black jacket planting vegetable in a empty abandoned garden.',
    'A black man in a black jacket sitting on a bench in a empty abandoned park.',
    'A black man in a black jacket stood beside the bench in panic in a empty abandoned park.',
    'A rabbit jumping in a empty abandoned park.',
    'A black man in a black jacket using a flashlight in a empty abandoned museum in dark.',
    'A black man in a black jacket holding a document in a empty abandoned museum in dark.',
    'A black man in a black jacket using binoculars on the rooftop in evening.',
    'A black man in a black jacket press a button in the empty abandoned room.', # x
    'Electricity flickered through the doors of the empty abandoned manor in the night.', # x
    'A black man in a black jacket using tools to fix black Armored SUV car in dilapidated garage.',
    'A black man in a black jacket shave with razors in a empty abandoned room.', # x
    'A black man in a black jacket looked at the map on the wall in a empty abandoned room.',
    'A black Armored SUV car traveling in an empty abandoned tunnel.',
    'A black man in a black jacket near a empty abandoned bus in empty abandoned tunnel.',
    'A black man in a black jacket holding a gasoline can near a black Armored SUV car in empty abandoned tunnel.', #x
    'A black Armored SUV car traveling in an empty abandoned freeway.',
    'A black man in a black jacket walking in an empty abandoned cinema.',
    'A black man in a black jacket watching a movie in an empty abandoned cinema.',
    'A black man in a black jacket drinking cola in an empty abandoned cinema.',
    'A black German Shepherd running in an empty abandoned cinema.',
    'A zombie attacking A black man in a black jacket in an empty abandoned cinema.',
    'A black man in a black jacket and A black German Shepherd running in an empty abandoned cinema.',
    'A black man in a black jacket and A black German Shepherd in a empty abandoned canteen.',
    'A black man in a black jacket injecting drugs in a empty abandoned canteen.',
    'A black man in a black jacket eating lunch in a empty abandoned canteen.',
    'A black German Shepherd siting near a door in an empty abandoned canteen.',
    'Many zombies near a empty abandoned canteen.',
    'A black man in a black jacket and A black German Shepherd walking in the empty abandoned canteen kitchen.', # x
    'A black man in a black jacket and A black German Shepherd in an narrow alley.',
    'A Mice in cages in a empty abandoned lab.',
    'A black man in a black jacket looking a medical book in a empty abandoned lab.',
    'An huge explosion in the empty abandoned chemical plant.',
    'A black German Shepherd running to the fired room in empty abandoned chemical plant.',
    'A black man in a black jacket holding a pharmacist in empty abandoned room.',
    'A black man in a black jacket marking on the map on the table in empty abandoned room.',
    'A black German Shepherd is lying by a A black man in a black jacket in empty abandoned room.',
    'A black man in a black jacket drinking bottled water in empty abandoned room.',
    'A black man in a black jacket walking in the empty abandoned barracks.',
    'A black man in a black jacket holding a gun in the empty abandoned barracks.',
    'A black German Shepherd barking in the empty abandoned barracks.',
    'A black man in a black jacket use gun shooting in the empty abandoned barracks.',
    'A zombie lying on the ground in empty abandoned barracks.',
    'A black man in a black jacket holding a tube in the empty abandoned barracks.', # x
    'A black man in a black jacket find a box of canned goods in the empty abandoned barracks.',
    'A black man in a black jacket lying in a flysheet in evening.',
    'A black man in a black jacket writing a diary in a flysheet in evening.',
    'A black man in a black jacket and A black German Shepherd walking in a lake in forest.',
    'A black man in a black jacket finding a kayaking in the lake in forest.',
    'A black man in a black jacket and A black German Shepherd on the kayaking floating on the river.', # x
    'A black man in a black jacket and A black German Shepherd walking on the empty beach.',
    'A black man in a black jacket look a flare in the empty beach.',
    'An aircraft carrier appears at sea level.',
    'A aircraft carrier near a empty beach.',
    'A black man in a black jacket and A black German Shepherd walking on the deck of A aircraft carrier.',
    'The aircraft carrier departed into the distance sea.'
]

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', default=r"./RealVisXL_V4.0", type=str)
parser.add_argument('--image_encoder_path', type=str, default=r"./IP-Adapter/sdxl_models/image_encoder")
parser.add_argument('--ip_ckpt', default=r"./IP-Adapter/sdxl_models/ip-adapter_sdxl.bin", type=str)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--story', default=story1, type=list)

args = parser.parse_args()

base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt
device = args.device

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# load SD pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None
)

seed = random.randint(0, 100000)
print(seed)

# load story-adapter
storyadapter = StoryAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

character=True
fixing_prompts = []
for prompt in args.story:
    if character == True:
        if 'Robinson' in prompt:
            prompt = prompt.replace('Robinson', 'a man, wearing tattered sailor clothes.')
        if 'Friday' in prompt:
            prompt = prompt.replace('Friday', 'a chimpanzee.')
    fixing_prompts.append(prompt)

prompts = fixing_prompts

os.makedirs(f'./story', exist_ok=True)
os.makedirs(f'./story/results_xl', exist_ok=True)


for i, text in enumerate(prompts):
    images = storyadapter.generate(pil_image=None, num_samples=1, num_inference_steps=50, seed=seed,
            prompt=text, scale=0.3, use_image=False)
    grid = image_grid(images, 1, 1)
    grid.save(f'./story/results_xl/img_{i}.png')

images = []
for y in range(len(prompts)):
    image = Image.open(f'./story/results_xl/img_{y}.png')
    image = image.resize((256, 256))
    images.append(image)


scales = np.linspace(0.3,0.5,10)
print(scales)

for i, scale in enumerate(scales):
    new_images = []
    os.makedirs(f'./story/results_xl{i+1}', exist_ok=True)
    print(f'epoch:{i+1}')
    for y, text in enumerate(prompts):
        image = storyadapter.generate(pil_image=images, num_samples=1, num_inference_steps=50, seed=seed,
                                  prompt=text, scale=scale, use_image=True)
        new_images.append(image[0].resize((256, 256)))
        grid = image_grid(image, 1, 1)
        grid.save(f'./story/results_xl{i+1}/img_{y}.png')
    images = new_images
