# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# This is a simple example for a custom action which utters "Hello World!"

from time import sleep
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from fastai.vision import load_learner, open_image, BytesIO
from prettytable import PrettyTable
from random import choice

from PIL import Image
from urllib.request import urlretrieve
from io import BytesIO
import requests
import json
from pprint import pprint
import os
from unidecode import unidecode

# PLANT ACTIONS ---------------------------------------------------------------

plant_names = ["acelga", "achicoria", "radicheta", "ajo", "albahaca", "apio", "arvejas", "batata", "berenjena", "brócoli", "calabaza", "zapallo", "caléndula", "cebolla", "cebolla de verdeo", "cebollino", "ciboulet", "choclo", "cilantro", "coliflor", "escarola", "espinaca", "frambuesa", "frutilla", "haba", "kale", "lechuga", "mora", "nabo", "papa", "pepino", "perejil", "pimiento", "poroto", "chaucha", "puerro", "quinoa", "rabanito", "remolacha", "repollito de bruselas", "repollo morado", "repollo blanco", "repollo verde", "rúcula", "tomate", "topinambur", "zanahoria", "zapallito de tronco"]


class ActionIdentifyPlant(Action):

    def name(self) -> Text:
        return "action_identify_plant"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        API_KEY = ""  # Set you API_KEY here
        api_endpoint = f"https://my-api.plantnet.org/v2/identify/all?lang=es&api-key={API_KEY}"

        img_url = tracker.latest_message["text"]
        img_unique_id = tracker.latest_message["metadata"]["message"]["photo"][-1]["file_unique_id"]

        print(img_url)
        img_filename = f"./actions/photos/identify_plant/{img_unique_id}.jpg"

        urlretrieve(img_url, img_filename)

        image_path_1 = img_filename
        image_data_1 = open(image_path_1, 'rb')

        data = {
            'organs': ['leaf']
        }

        files = [
            ('images', (image_path_1, image_data_1))
        ]

        pprint(f'requesting {api_endpoint}')
        req = requests.Request('POST', url=api_endpoint,
                               files=files, data=data)
        prepared = req.prepare()

        s = requests.Session()
        response = s.send(prepared)
        pprint(f"status code is {response.status_code}")

        if response.status_code != 200:
            return [SlotSet("plant_name", "planta_desconocida")]

        json_result = json.loads(response.text)
        # pprint(json_result)

        result = json_result['results'][0]
        common_names_list = result['species']['commonNames']
        common_name_list_lowecase = [name.lower()
                                     for name in common_names_list]
        scientific_name = result['species']['scientificName']
        gbif_id = result['gbif']['id']
        confidence_score = str(int(100*round(result['score'], 2)))
        gbif_url = f'https://www.gbif.org/es/species/{gbif_id}/metrics'
        plant_net_url = f'https://identify.plantnet.org/es/the-plant-list/species/{scientific_name.replace(" ", "%20")}/data'
        print(confidence_score)

        common_name = ""
        for name in common_name_list_lowecase:
            if name in plant_names:
                common_name = name
                break

        if common_name != "":
            dispatcher.utter_message(
                f'Identifico que es una planta de {common_name} con una confianza de {confidence_score}%.')
            dispatcher.utter_message(
                f'Estos son algunos recursos útiles de esta planta:\n{plant_net_url}\n{gbif_url}')
            print("setteando slot a " + common_name)
            return [SlotSet("plant_name", common_name)]
        elif len(common_names_list) > 0:
            dispatcher.utter_message(
                f'Identifico que es una planta de {common_names_list[0]} con una confianza de {confidence_score}%.')
        else:
            dispatcher.utter_message(
                f'Identifico que es una planta de {scientific_name} con una confianza de {confidence_score}%.')

        dispatcher.utter_message(
            f'Estos son algunos recursos útiles de esta planta:\n{plant_net_url}\n{gbif_url}')
        # dispatcher.utter_message(gbif_url)
        # dispatcher.utter_message(plant_net_url)

        return [SlotSet("plant_name", "planta_desconocida")]

recipe_image_database = {
    "acelga": ["https://imgur.com/jsWLbuC.png", "https://imgur.com/zL3dgll.png", "https://imgur.com/Sgs2gN6.png"],
    "arvejas": ["https://imgur.com/Gy1PxuA.png"],
    "cilantro": ["https://imgur.com/3EAuwZh.png"],
    "coliflor": ["https://imgur.com/cQ2ThNN.png"],
    "espinaca": ["https://imgur.com/RjYqAoF.png", "https://imgur.com/gSVvtbP.png", "https://imgur.com/BRskVsM.png"],
    "frambuesa": ["https://imgur.com/XcWdPJJ.png", "https://imgur.com/yeb63s6.png"],
    "frutilla": ["https://imgur.com/317I1UA.png", "https://imgur.com/0wJj647.png"],
    "haba": ["https://imgur.com/xseHGLH.png", "https://imgur.com/jpQ0Pft.png", "https://imgur.com/8v9DfFQ.png", "https://imgur.com/Tlp6lAj.png"],
    "kale": ["https://imgur.com/EUPmcz8.png"],
    "pepino": ["https://imgur.com/4Jnl4IH.png"],
    "pera": ["https://imgur.com/P7qTLZJ.png"],
    "perejil": ["https://imgur.com/RixhWjW.png"],
    "repollo blanco": ["https://imgur.com/piwFiAK.png", "https://imgur.com/xPxnQHw.png"],
    "tomate": ["https://imgur.com/eBaoDDI.png", "https://imgur.com/gZrf4RN.png"],
    "zanahoria": ["https://imgur.com/VFWETDi.png"],
    "zapallito de tronco": ["https://imgur.com/7qHgdTj.png", "https://imgur.com/mSsKegF.png"]
}

class ActionProvideSlottedPlantRecipe(Action):

    def name(self) -> Text:
        return "action_provide_slotted_plant_recipe"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        plant_name = unidecode(tracker.get_slot("plant_name").lower())

        if plant_name in recipe_image_database:
            dispatcher.utter_message(image=choice(recipe_image_database[plant_name]))
        else:
            dispatcher.utter_message(
                text="Todavía no tenemos recetas de " + plant_name)

        return []


# nutritional_info_database = {
#     "acelga": ["Es una verdura con un gran contenido de agua, por lo que es rica en ciertas vitaminas, sales minerales y agua. Si la ingerimos con regularidad nos va a ayudar en la formación de anticuerpos del sistema inmunológico, ya que tiene vitamina A, esencial para la visión, el buen estado de la piel, el cabello y las mucosas. Dentro de sus propiedades se destaca por su mayor contenido en magnesio, sodio, yodo, hierro y calcio.", "Es una verdura rica en ciertas vitaminas, sales minerales y agua, destacándose por su gran contenido en magnesio, sodio, yodo, hierro y calcio. Ingerir regularmente esta verdura nos va a ayudar en la formación de anticuerpos del sistema inmunológico, ya que tiene vitamina A, esencial para la visión, el buen estado de la piel, el cabello y las mucosas."],
#     "espinaca": ["Las espinacas destacan sobre todo por su riqueza en vitaminas y minerales. La provitamina A es esencial para la visión, el buen estado de la piel, el cabello, las mucosas, los huesos y para el buen funcionamiento del sistema inmunológico. La vitamina C interviene en la formación de colágeno, glóbulos rojos, huesos y dientes, al tiempo que favorece la absorción del hierro de los alimentos y aumenta la resistencia frente a las infecciones.", "Las espinacas destacan principalmente por su riqueza en vitaminas y minerales. La provitamina A es esencial para la visión, el buen estado de la piel, el cabello, las mucosas, los huesos y para el buen funcionamiento del sistema inmunológico. La vitamina C interviene en la formación de colágeno, glóbulos rojos, huesos y dientes, al tiempo que favorece la absorción del hierro de los alimentos y aumenta la resistencia frente a las infecciones."],
#     "kale": ["El kale es un vegetal perteneciente a la familia de las coles. Entre las propiedades del kale se destaca,  además de su bajo valor calórico debido a que posee una elevada proporción de agua en su composición, la fibra y su riqueza en minerales. El kale posee calcio, hierro, magnesio, potasio.\nEl potasio, es un mineral necesario para la transmisión y generación del impulso nervioso y para la actividad muscular normal.", "El kale es un vegetal perteneciente a la familia de las coles. Algunas de las propiedades del kale las cuales destacan, además de su bajo valor calórico debido a que posee una elevada proporción de agua en su composición, la fibra y su riqueza en minerales. El kale posee calcio, hierro, magnesio, potasio.\nEl potasio, es un mineral necesario para la transmisión y generación del impulso nervioso y para la actividad muscular normal."],
#     "arvejas": ["Las arvejas son ricas en proteínas y carbohidratos, bajas en grasa y constituyen una buena fuente de fibra, vitaminas complejo B y C; cuando se consumen frescas o secas enteras o partidas. La fibra de la arveja es soluble en agua, promueve el buen funcionamiento intestinal.", "Las arvejas son ricas en carbohidratos y proteínas, bajas en grasa y además de una buena fuente de fibra, vitaminas B y C; cuando se consumen frescas o secas enteras o partidas. La fibra de la arveja es soluble en agua, promueve el buen funcionamiento intestinal."],
#     "haba": ["Las habas son un alimento con gran contenido en agua, son ricas en hidratos de carbono complejos y fibra. Si algo destaca en las habas es el contenido de fibra y de ácido fólico. Las fibras contribuyen a regular el tránsito intestinal.", "Las habas son un alimento con gran contenido en agua, son ricas en hidratos de carbono complejos y fibra. Si algo destaca en las habas es el contenido de fibra, las cuales contribuyen a regular el tránsito intestinal y de ácido fólico."],
#     "pepino": ["Es una hortaliza de bajo aporte calórico debido a su reducido contenido en hidratos de carbono, en comparación con otras hortalizas, y a su elevado contenido de agua. Aporta fibra, pequeñas cantidades de vitamina C, provitamina A y vitamina E, y, en proporciones aún menores, vitaminas del grupo B tales como folatos, B1, B2 y B3. La vitamina A es esencial para la visión, el buen estado de la piel, el cabello, las mucosas, los huesos y para el buen funcionamiento del sistema inmunológico.", "Es una hortaliza de bajo aporte calórico debido a su reducido contenido en hidratos de carbono, en comparación con otras hortalizas, y a su elevado contenido de agua. Aporta fibra, pequeñas cantidades de vitamina C, provitamina A y vitamina E, y, en proporciones aún menores, vitaminas del grupo B tales como folatos, B1, B2 y B3. La vitamina A es esencial para la visión, el buen estado de la piel, el cabello, las mucosas, los huesos y para el buen funcionamiento del sistema inmunológico."],
#     "zapallito de tronco": ["Además de sus valores nutricionales y su escaso aporte de calorías, el zapallito es un alimento ideal en cualquier dieta para adelgazar, ya que contiene fibra que ayuda a depurar el organismo, estimulando el peristaltismo intestinal."],
#     "frambuesa": ["La frambuesa posee importantes nutrientes como la vitamina C, útil para la absorción de hierro.\nEl magnesio, hierro y fósforo son los minerales predominantes en la frambuesa, por lo que es un alimento recomendado para las personas con hipertensión arterial o afecciones de los vasos sanguíneos y del corazón.", "La frambuesa posee importantes nutrientes como la vitamina C, útil para la absorción de hierro, por lo que es un alimento recomendado para las personas con hipertensión arterial o afecciones de los vasos sanguíneos y del corazón.\nEl magnesio, hierro y fósforo son los minerales predominantes en la frambuesa, por lo que es un alimento recomendado para las personas con hipertensión arterial o afecciones de los vasos sanguíneos y del corazón.", "Además de contener una amplia variedad de antioxidantes, que desempeñan un papel importante en la prevención de enfermedades cardiovasculares y ciertos tipos de cáncer, la frambuesa es una buena fuente de fibra y se le atribuyen propiedades diuréticas.\nTambién contiene ácido fólico, imprescindible para las embarazadas, un elemento que interviene en la producción de glóbulos rojos y blancos y en la formación de anticuerpos del sistema inmunológico."],
#     "frutilla": ["La frutilla aporta pocas calorías, una gran cantidad de agua (lo que facilita la hidratación de nuestro cuerpo), fibras, azúcares, vitamina y minerales. Presenta mucha vitamina C y otros antioxidantes en contra de los radicales libres, los principales causantes de diversos tipos de cáncer y el envejecimiento.\nTambién facilita la menor absorción de carbohidratos, así como mejora el tránsito intestinal, contribuyendo de este modo al mantenimiento de los niveles de azúcar en sangre."],
#     "mora": ["Una de las principales características de la mora es que aporta una buena cantidad de fibra. Ayuda a reducir el colesterol, mejora la digestión y mejora la salud del colon.\nAsimismo, las moras contienen numerosas vitaminas como las vitaminas A, C, E, K y el ácido fólico. La vitamina A nos ayuda con nuestra salud ocular. La vitamina C mejora la inmunidad y nos proporciona antioxidantes esenciales. La vitamina E actúa como un antioxidante y combate los radicales libres.", "Una de las principales características de la mora es que aporta una buena cantidad de fibra. Ayuda a reducir el colesterol, mejora la digestión y mejora la salud del colon.\nLas moras contienen numerosas vitaminas como las vitaminas A, C, E, K y el ácido fólico. La vitamina A nos ayuda con nuestra salud ocular. La vitamina C mejora la inmunidad y nos proporciona antioxidantes esenciales. La vitamina E actúa como un antioxidante."],
#     "tomate": ["El tomate tiene gran cantidad de fibra, minerales como el potasio y el fósforo, y de vitaminas, entre las que destacan la C, E, provitamina A y vitaminas del grupo B, en especial B1 y niacina o B3. Además, presenta un alto contenido en carotenos como el licopeno, pigmento natural que aporta al tomate su color rojo característico. El alto contenido en vitaminas C y E y la presencia de carotenos en el tomate convierten a éste en una importante fuente de antioxidantes, sustancias con función protectora de nuestro organismo.\nLa vitamina E, al igual que la C, tiene acción antioxidante, y ésta última además interviene en la formación de colágeno, glóbulos rojos, huesos y dientes. También favorece la absorción del hierro de los alimentos y aumenta la resistencia frente a las infecciones.\nLa vitamina A es esencial para la visión, el buen estado de la piel, el cabello, las mucosas, los huesos y para el buen funcionamiento del sistema inmunológico, además de tener propiedades antioxidantes.\nLa niacina o vitamina B3 actúa en el funcionamiento del sistema digestivo, el buen estado de la piel, el sistema nervioso y en la conversión de los alimentos en energía.\nEl potasio es un mineral necesario para la transmisión y generación del impulso nervioso y para la actividad muscular normal, además de intervenir en el equilibrio de agua dentro y fuera de la célula."],
#     "coliflor": ["Es un alimento de baja densidad energética, ya que está compuesto principalmente por agua, con un bajo contenido de hidratos de carbono, proteínas y lípidos.\nSe destaca la presencia de vitamina C, cuya principal función es contribuir a la protección de las células frente al daño oxidativo y mejorar la absorción del hierro. También tiene folatos, que contribuyen a la formación normal de las células sanguíneas y al funcionamiento adecuado del sistema inmunitario.\nEn cuanto a su contenido de minerales, este vegetal es fuente de potasio, que contribuye al funcionamiento normal del sistema nervioso y de los músculos, además del mantenimiento de la tensión arterial normal."],
#     "repollo blanco": ["Es un alimento rico en vitamina C y folatos. Aporta cantidades apreciables de potasio, hierro, fósforo y, en menor cantidad, de calcio. Por su contenido en fibra (soluble e insoluble), favorece al tránsito intestinal y ayuda a combatir el estreñimiento.\nContiene fitonutrientes, que le confieren propiedades preventivas sobre diversos tipos de cáncer."]
# }

nutritional_info_database = {
    "acelga": "https://imgur.com/thhQCwC.png",
    "espinaca": "https://imgur.com/o22GTwS.png",
    "kale": "https://imgur.com/PokNLYV.png",
    "arvejas": "https://imgur.com/BsSkx9V.png",
    "haba": "https://imgur.com/IA6wEbk.png",
    "pepino": "https://imgur.com/GoYOu1N.png",
    "zapallito de tronco": "https://imgur.com/mJH5ppK.png",
    "frambuesa": "https://imgur.com/yuKZCVG.png",
    "frutilla": "https://imgur.com/eEFzO6l.png",
    "mora": "https://imgur.com/z2SC6S7.png",
    "tomate": "https://imgur.com/zpVfFoM.png",
    "coliflor": "https://imgur.com/Fskd8gR.png",
    "repollo blanco": "https://imgur.com/dNfOCbL.png",
    "cilantro": "https://imgur.com/s4s2dn1.png",
    "pera": "https://imgur.com/nHX4m0A.png"
}


class ActionProvideSlottedPlantNutricionalInfo(Action):

    def name(self) -> Text:
        return "action_provide_slotted_plant_nutritional_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        plant_name = unidecode(tracker.get_slot("plant_name").lower())

        if plant_name in nutritional_info_database:
            dispatcher.utter_message(image=nutritional_info_database[plant_name])
        else:
            dispatcher.utter_message(
                text=f"Todavía no tenemos información nutricional de {plant_name}")
        
        return []

nurture_info_image_database = {
    "acelga": "https://imgur.com/u49FLhg.png",
    "achicoria": "https://imgur.com/K5eLYyZ.png",
    "radicheta": "https://imgur.com/K5eLYyZ.png",
    "ajo": "https://imgur.com/ypYXdl6.png",
    "albahaca": "https://imgur.com/Ftu0AFo.png",
    "apio": "https://imgur.com/EeV32X3.png",
    "arvejas": "https://imgur.com/UMV1ooi.png",
    "batata": "https://imgur.com/yYvW6vO.png",
    "berenjena": "https://imgur.com/6eFhBCF.png",
    "brócoli": "https://imgur.com/W5kv6YJ.png",
    "calabaza": "https://imgur.com/95AfVpV.png",
    "zapallo": "https://imgur.com/95AfVpV.png",
    "calendula": "https://imgur.com/gihUIQk.png",
    "cebolla": "https://imgur.com/luT9FmX.png",
    "cebollita de verdeo": "https://imgur.com/KwQYfqb.png",
    "cebollino": "https://imgur.com/Fx82uZp.png",
    "ciboulet": "https://imgur.com/Fx82uZp.png",
    "choclo": "https://imgur.com/Fc8kEsZ.png",
    "cilantro": "https://imgur.com/5D6pJfA.png",
    "coliflor": "https://imgur.com/KOKiXOy.png",
    "escarola": "https://imgur.com/71cRgIe.png",
    "espinaca": "https://imgur.com/NBgiprG.png",
    "frambuesa": "https://imgur.com/vTAqhfH.png",
    "frutilla": "https://imgur.com/0HVOz0c.png",
    "haba": "https://imgur.com/nK06jDd.png",
    "kale": "https://imgur.com/UHsITmy.png",
    "lechuga": "https://imgur.com/r8WnXHV.png",
    "mora": "https://imgur.com/7fbmGp9.png",
    "nabo": "https://imgur.com/DI1rj3Y.png",
    "papa": "https://imgur.com/ExarlBJ.png",
    "pepino": "https://imgur.com/VoNwCtt.png",
    "perejil": "https://imgur.com/xeMVsea.png",
    "pimiento": "https://imgur.com/SmtBLKa.png",
    "poroto": "https://imgur.com/RwpX7y7.png",
    "chaucha": "https://imgur.com/RwpX7y7.png",
    "puerro": "https://imgur.com/siXBV9s.png",
    "quinoa": "https://i.imgur.com/lCzXtT9.png",
    "rabanito": "https://imgur.com/nxB2GpM.png",
    "remolacha": "https://imgur.com/0JXGKKV.png",
    "repollito de bruselas": "https://imgur.com/iocE8Va.png",
    "repollo morado": "https://imgur.com/60XgvXp.png",
    "repollo blanco": "https://imgur.com/60XgvXp.png",
    "repollo verde": "https://imgur.com/60XgvXp.png",
    "rucula": "https://imgur.com/MqsUwit.png",
    "tomate": "https://imgur.com/pmO7hvz.png",
    "topinambur": "https://imgur.com/8l8ymu3.png",
    "zanahoria": "https://imgur.com/o5Say71.png",
    "zapallito de tronco": "https://imgur.com/hTgcDkS.png"
}

class ActionProvideSlottedPlantNurtureInfo(Action):

    def name(self) -> Text:
        return "action_provide_slotted_plant_nurture_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        plant_name = unidecode(tracker.get_slot("plant_name").lower())
        if plant_name in nurture_info_image_database:
            dispatcher.utter_message(image=nurture_info_image_database[plant_name])
        else:
            dispatcher.utter_message(
                text="Todavía no tenemos la ficha de cultivo de " + plant_name)

        return []

# INSECT ACTIONS --------------------------------------------------------------
class ActionIdentifyInsect(Action):

    def name(self) -> Text:
        return "action_identify_insect"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        learn = load_learner("./actions/",'10insectos_fastaiV1.pkl')

        img_url = tracker.latest_message["text"]
        img_unique_id = tracker.latest_message["metadata"]["message"]["photo"][-1]["file_unique_id"]

        img_filename = f"./actions/photos/identify_insect/{img_unique_id}.jpg"

        urlretrieve(img_url, img_filename)
        
        with open(img_filename, "rb") as image:
            img_bytes = image.read()
            img = open_image(BytesIO(img_bytes))
            prediction = learn.predict(img)
            prediction_str = str(prediction[0])
            confidence_top_prediction = max([round(p,3) for p in prediction[2].numpy()])
            dispatcher.utter_message(text=f"Identifico que es {prediction_str} con una confianza del {round(confidence_top_prediction*100, 4)}%")

        # x = PrettyTable()
        # x.field_names = ["Label", "Confidence"]

        # classes = ['babosa', 'babosita_del_peral', 'caracol', 'chinche_verde', 'cochinilla', 'hormiga', 'mosca_blanca', 'oruga', 'pulgon', 'tijereta']
        # print(f"Prediction: {prediction[0]}\n")
        # confidence_scores = [round(p,3) for p in prediction[2].numpy()]
        # x.add_rows(sorted(zip(classes, confidence_scores), key=lambda x:x[1], reverse=True))
        # dispatcher.utter_message(json_message={
        #     "text": f"```{x.get_string()}```",
        #     "parse_mode": 'MarkdownV2'
        # })
        return [SlotSet("insect_name", prediction_str)]

insect_what_database = {
    "babosita del peral": ["La babosita del peral, a pesar de su nombre, es una avispa. Considerada como una de las plagas más importantes de los árboles frutales en la Patagonia.", "La babosita del peral es una avispa que en su estado larval posee una cubierta mucilaginosa que le otorga aspecto de babosa."],
    "tijereta": ["Son insectos que se han convertido en una plaga doméstica y de cultivos. Es originaria de Europa, Asia Occidental y África.", "Son insectos que se alimentan de plantas cuando están en grandes poblaciones. Pueden ser benéficos, pues se alimentan de otros insectos, como pulgones, ácaros y pequeñas arañas. Incluso se alimentan de materia en descomposición, por eso podemos encontrarlo en el compost."],
    "babosa": ["Las babosas son invertebrados moluscos (no son insectos). Las babosas se alimentan de plantas, líquenes y hongos pero también pueden consumir residuos animales.", "Las babosas son invertebrados que suelen considerarse un problema en jardines y huertas, debido a que se alimentan de varias especies de plantas cultivadas por el hombre."],
    "hormiga": ["Las hormigas son insectos que pertenecen a un grupo llamado 'insectos sociales', los cuales se caracterizan por vivir en comunidades organizadas con división de tareas. El tamaño de las colonias de hormigas pueden variar desde una docena hasta varios miles de ejemplares.", "Las hormigas son insectos que depende su alimentación pueden ser nectívoras (se alimentan de soluciones azucaradas), granívoras (se alimentan de granos y semillas) y cortadoras (recolectan trozos de hojas que trasladan al nido para cultivar un hongo que sirve de alimento de adultos y larvas).", "Las hormigas son insectos, que cumplen roles importantes en los ecosistemas, entre ellos, regular el crecimiento vegetal, reciclar nutrientes, ser controladores biológicos, polinizar, remover materia orgánica..."],
    "oruga": ["Las orugas son larvas de mariposas, polillas o cascarudos. Que se alimentan de hojas, flores, tallos, frutos o raíces."],
    "pilme": ["El pilme es un insecto que se alimenta del follaje de distintas plantas para el cultivo, así como de otras plantas silvestres . Las larvas de pilmes son parásitos obligados y voraces comedoras de los huevos de langostas.", "Los pilmes (Epicauta) son insectos terrestres, de movimientos lentos, y realizan vuelos solo cuando son exigidos."],
    "pulgon": ["Los pulgones son insectos chupadores que se alimentan de la savia de los tallos y las hojas de las plantas. Pueden aparecer en el envés de las hojas y en los brotes tiernos.","Los pulgones son parásitos que constituyen plagas que comprometen el valor de los cultivos, así también como a las plantas ornamentales. ","Los pulgones son parásitos de plantas que se alimentan de la savia de frutales, hortalizas ornamentales y especies de valor forestal."],
    "mosca_blanca":["Aparece cuando hace calor, en general en el tomate. Es fácil detectarla porque al mover la planta salen volando, son muy pequeñas con cuerpo amarillo y alas blancas, que se encuentran en el envés o parte de debajo de la hoja."],
    "sirfido":["Son insectos benéficos, emparentados con las moscas, tienen vuelos suspendidos, como los colibríes. En estado adulto son polinizadores y como larvas se alimentan de pulgones y cochinillas."],
    "vaquita": ["Son insectos benéficos, emparentados con los cascarudos, como adultos ponen huevos amarillos y organizados sobre las hojas. Luego nacen las larvas que parecen pequeños cocodrilos negros, la cual se alimentan de pulgones, luego tienen un estadio inmóvil, llamado pupa. Y por último, y completando su ciclo, se transforman en adulto que también se alimentan de pulgones."],
    "cochinilla": ["Presenta formas muy diferentes, son insectos sedentarios en estado adulto, sin alas, algunas tienen un caparazón pequeño inmóvil, otras tienen pelos blancos que le da aspecto algodonoso y se adhieren a ramas, tallos y hojas. "]
}

insect_physical_appearence_database = {
    "babosita del peral": ["Un adulto es una avispa de color oscuro aproximadamente de 8 mm de largo. Su cuerpo es ancho, con el abdomen ampliamente unido al tórax.", "Las babositas del peral adultas son insectos oscuros que poseen ojos y antenas negras y dos pares de alas anchas y transparentes."],
    "tijereta": ["Normalmente, los individuos adultos son de color castaño oscuro con vetas color canela. Y miden entre 1 y 2.5cm. Los adultos pueden ser alados o carecer de alas.", "Los machos son más grandes que las hembras y tienen un par de pinzas robustas en la punta del abdomen (fórceps). Las hembras son de color más claro y sus fórceps son más pequeños."],
    "babosa": ["Las babosas no poseen caparazón o, si lo tienen es pequeño e interno.", "El cuerpo de las babosas terrestre es alargado y mide entre 1 y 15 cm . La cabeza tiene dos pares de antenas. El par superior es sensible a la luz, y el inferior provee el sentido del olfato.", "Las babosas poseen un manto (o escudo) detrás de la cabeza. Este es una laminilla calcárea que cubre algunos órganos.", "El cuerpo de las babosas posee una importante cantidad de agua, y se encuentra recubierto por un mucus protector que evita la desecación y facilita el desplazamiento."],
    "hormiga": ["La longitud normal de una hormiga es de 1 a 5 mm, aunque se han llegado a descubrir hormigas de hasta 30 mm (no en la región patagónica).", "Tienen cabeza grande, antenas articuladas, poderosas mandíbulas y tres regiones corporales (cabeza, tórax y abdomen). En el extremo último del abdomen se puede encontrar el aguijón."],
    "oruga": ["Las orugas poseen una estructura blanda y cilíndrica, excepto la cabeza. La cabeza suele ser una cápsula resistente y dura, en la que se insertan dos potentes mandíbulas.","El cuerpo de la oruga se encuentra dividido en una serie de segmentos, los cuales presentan tres pares de patas más varios pares de falsas patas que usan para caminar y agarrarse.","Algunas especies de orugas suelen exhibir en su cuerpo una variedad de colores o espinas que usan para anunciar su toxicidad o desagradable sabor como mecanismo de defensa ante predadores."],
    "pilme": ["El pilme tiene un cuerpo alargado de entre 9 a 14mm, y es color negro brillante. Se le reconoce por sus antenas largas y negras. Los extremos superiores de sus patas son de color anaranjado-rojizo."],
    "pulgon": ["Son pequeños, de colores variados, en general, verdes amarillos o negros. Suelen ser lisos pero también pueden tener manchas. Pueden o no tener alas. Además tienen en el abdomen dos tubitos o sifones que segregan un líquido azucarado y pegajoso, denominado melaza, que impregna la superficie de la planta."]
}

insect_habitat_database = {
    "babosita del peral": ["Las babositas del peral son nativas de Europa, y se encuentran presentes en Uruguay, Chile y Argentina. Las larvas, quienes consumen el tejido de las hojas, están presentes entre finales de diciembre y hasta principios de Marzo."],
    "tijereta": ["Las tijeretas prefieren áreas húmedas y oscuras. Son insectos de hábitos nocturnos y buscan protección durante el día.", "Sus sitios de refugio son lugares oscuros y tranquilos, como grietas, cavidades, bajo piedras, leña, enmalezados o camellones con mulching.", "Las tijeretas prefieren sitios exteriores a menos que existan poblaciones excesivamente grandes o condiciones ambientales adversas. Y en estos casos suelen introducirse en sótanos, bodegas, invernáculos o sitios donde encuentren alimentos."],
    "babosa": ["Las babosas viven en ambientes húmedos y desarrollan sus actividades durante la noche. Están activas durante temperaturas entre 0°C y 18°C. Durante el día las babosas buscan refugio bajo ladrillos, piedras, tablones de madera, entre otros."],
    "hormiga": ["Las hormigas habitan en casi todos los ecosistemas terrestres con excepción de la Antártida. Las especies que entran a nuestras casas en busca de protección o alimento son conocidas como 'Hormigas Urbanas'.", "Las hormigas viven en nidos o colonias de tamaños variables, desde una docena hasta varios miles de individuos."],
    "oruga": ["Hay aproximadamente 30 especies de orugas que se distribuyen a lo largo de seis continentes. En América se encuentra en varios países desde Argentina y Chile hasta el sur de Estados Unidos."],
    "pilme": ["Los pilmes se distribuyen en Chile y Argentina, desde el norte de Neuquén al centro de Chubut, principalmente en la zona cordillerana. El adulto aparece en la primavera, a partir de octubre y permanece activo durante el verano hasta febrero o marzo."],
    "pulgon": ["Mayormente están distribuidos en zonas templadas, y atacan a las plantas principalmente durante la primavera y el verano, aunque durante el invierno pueden encontrarse en plantas dentro de los invernaderos.", "Se encuentran particularmente favorecidos por la sequedad ambiental y el exceso de fertilizantes.", "Los pulgones aparecen principalmente en primavera y verano, cuando hay brotes nuevos en las plantas. Seleccionan preferentemente plantas como frutales, plantas ornamentales y hortalizas. En particular, el duraznero, los rosales, el repollo y kale son plantas especialmente sensibles a ser afectados por pulgones."]
}

class ActionProvideSlottedInsectGeneralInfo(Action):

    def name(self) -> Text:
        return "action_provide_slotted_insect_general_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        insect_name = unidecode(tracker.get_slot("insect_name").lower())

        if insect_name not in insect_what_database:
            dispatcher.utter_message(
                text=f"Todavía no tenemos información general de {insect_name}")
            return []
        
        if insect_name in insect_what_database:
            dispatcher.utter_message(text=choice(insect_what_database[insect_name]))

        if insect_name in insect_physical_appearence_database:
            dispatcher.utter_message(text=choice(insect_physical_appearence_database[insect_name]))

        if insect_name in insect_habitat_database:
            dispatcher.utter_message(text=choice(insect_habitat_database[insect_name]))

        return []

insect_harm_info_database = {
    "babosita del peral": ["Atacan principalmente el peral, y luego el cerezo, guindo, ciruelo, manzano y rara vez el duraznero. Los daños ocasionados son consecuencia de la alimentación de sus larvas.", "El daño que provocan las larvas hace que las hojas se mueran y queden retenidas en el árbol, el cual toma un aspecto de árbol seco.", "El daño provocado por la babosita del peral se debe principalmente a la alimentación del tejido de las hojas, dejando solamente las nervaduras y la epidermis inferior, y provocando una disminución de la capacidad fotosintética de la planta."],
    "tijereta": ["Las tijeretas son insectos inofensivos; aún así, los fórceps pueden producir la sensación de un pequeño pellizco. Generalmente, las tijeretas no son destructivas, ni venenosas y no morderán ni picarán a los humanos.", "Al momento de alimentarse, las tijeretas pueden generar agujeros profundos en los frutos, provocando su desvalorización y exponiéndolos al contacto directo con hongos u otros insectos.", "Cuando las poblaciones de tijeretas son abundantes, pueden alimentarse de plantas blandas dañando lechugas, fresas, dalias, margaritas y rosas."],
    "babosa": ["La mayoría de las especies de babosas son inofensivas. Sin embargo, un pequeño número de especies son plagas polífagas para la agricultura."],
    "hormiga": ["Las hormigas pueden picar o morder a los residentes y mascotas del hogar, y al picar pueden causar hinchazón, picazón o reacciones alérgicas en algunos casos.", "Las hormigas actúan como vectores de agentes patógenos. Dado su pequeño tamaño y su gran capacidad de desplazamiento, pueden propagar patógenos a los alimentos y a lugares sensibles como hospitales.", "Las hormigas pueden dañar plantas en jardines y huertas, al remover el suelo durante la confección de sus nidos, y al consumir frutos cultivados.", "Las hormigas pueden dañar aparatos electrónicos y materiales estructurales, tales como maderas en vigas o ventanas, revestimientos y cimientos."],
    "oruga": ["Las larvas, orugas o “gusanos” de cascarudos (coleópteros) se alimentan de raíces y tallos.", "Las larvas de mariposas y polillas (lepidópteros) se alimentan de hojas, flores y frutos.. Se observan sus larvas sobre las hojas, o en el envés de estas, verdes, marrones o negras. otras veces solo se observa el daño, hojas comidas, afectando la capacidad fotosintética de la planta si el ataque es intenso."],
    "pilme": ["Sobre la superficie del cuerpo, el pilme posee una sustancia aceitosa llamada Cantaridina que es irritable para las personas que tienen contacto con este insecto.", "El adulto se alimenta del follaje de diversas plantas de cultivo hortícola, dejando solo la nervadura central visible. Al reducir el área de la hoja, interrumpe la fotosíntesis y provoca la reducción en el crecimiento y calidad de los cultivos.", "Los principales cultivos afectados por el pilme son la papa, la alcachofa, ají, brócoli, espárragos, frutilla. alfalfa, betarraga, porotos, haba, zanahoria, tomate, trébol blanco y trébol rosado."],
    "pulgon": ["Los pulgones extraen nutrientes de la planta y alteran el balance de las hormonas del crecimiento. Esto debilita las plantas, detiene el crecimiento y hasta puede secarlas. Los pulgones excretan el exceso de azúcar, que al depositarse sobre las hojas favorece el desarrollo de un hongo negro llamdo fumagina. Y, además pueden transmitir virus y enfermar las plantas."],
    "mosca_blanca": ["Se alimenta de la savia de la planta y la debilita. Produce una melaza que atrae hormigas y hongos."],
    "sirfido": ["No provocan daño en las plantas. Son beneficiosos cuando aparecen en la huerta."],
    "vaquita": ["No provocan daño en las plantas. Son beneficiosos cuando aparecen en la huerta."],
    "cochinilla": ["Cada especie de cochinilla es especifica de una planta en particular, no se contagia a otras cercanas. Se alimentan de la savia de la planta y la debilitan. En algunos casos produce un liquido azucarado que alimenta a las hormigas y predispone al ataque de hongos."]
}

class ActionProvideSlottedInsectHarmInfo(Action):

    def name(self) -> Text:
        return "action_provide_slotted_insect_harm_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        insect_name = unidecode(tracker.get_slot("insect_name").lower())
        if insect_name in insect_harm_info_database:
            dispatcher.utter_message(
                text=choice(insect_harm_info_database[insect_name]))
        else:
            dispatcher.utter_message(
                text=f"Todavía no tenemos información de daño de {insect_name}")
        return []

insect_prevent_info_database = {
    "babosita del peral": ["Como método para el control de las larvas que invernan en el suelo,la remoción del suelo por debajo de la copa del árbol a 10 o 15 cm es otra manera de disminuir el ataque del insecto.","Existen agroquímicos que no se utilizan en agroecología y es arriesgado en huerta familiar porque tienen tiempo de carencia, o sea la fruta no se puede consumir inmediatamente, luego de la aplicación"],
    "tijereta": ["Para prevenir la presencia de las tijeretas se sugiere limpiar los jardines y lugares cercanos a las huertas o viviendas, para reducir la disponibilidad de lugares de refugio.", "Para prevenir la presencia de las tijeretas se puede esparcir cenizas a modo de repelente, alrededor de los invernáculos.", "Para prevenir las tijeretas se existe un repelente casero que se rocía a las plantas, preparado con: 5 dientes de ajo, 1 taza de agua y 3 gotas de jabón líquido. Se licua el ajo con el agua, se cuela, se agrega el jabón y se agrega agua hasta completar un litro de la mezcla.", "Se pueden construir trampas para las tijeretas: Pueden ser cartulina acanalada, ladrillos, trapos húmedos,... que sirvan como refugio. Las mismas deben colocarse a la tarde y recogerse por las mañanas cuando se coloca la trampa dentro de un recipiente con agua y detergente para matar a las tijeretas.", "Para controlar las tijeretas se pueden construir trampas que funcionan como refugios y que tengan algunos cebos, como una lata con aberturas pinchadas en los extremos, llena de cerveza y enterrada al ras del suelo. Este preparado las atrae, ahogándolas al caer dentro.", "Para controlar las tijeretas se pueden construir trampas que deben ser colocadas en el momento de mayor abundancia de tijeretas fuera del nido, aproximadamente a mediados del verano, en los meses de enero-febrero."],
    "babosa": ["Para prevenir la presencia de babosas se deben mantener los jardines libres de zonas con malezas o desechos vegetales que les ofrezcan refugio.", "Para prevenir la presencia de babosas es común el cultivo de especies repelentes como los geranios, begonias, capuchinas, salvia, romero y la lavanda alrededor de los canteros.", "Un método de control casero es colocar en el jardín un tablero elevado o una caja que les pueda servir de refugio, y hacer una limpieza diaria para eliminarlas.", "Un método de control casero es enterrar un pequeño recipiente lleno de agua o cerveza, la humedad y olor las atraerá y se ahogarán.", "Una opción para el control de babosas es colocar tierra de diatomeas, cáscaras de huevo trituradas o sal alrededor de las plantas que se desean proteger.", "Una opción para el control de babosas es colocar un papel de cobre o aluminio alrededor de las plantas o macetas.", "El método de control más utilizado es el uso de cebos compuestos de fosfato de hierro. Los cebos se colocan en lugares húmedos, al atardecer en días cálidos cuando las babosas están más activas."],
    "hormiga": ["El control definitivo de las colonias es aquel que busque eliminar a su reina. Pero en general es difícil dado que solo durante la fundación de la colonia la reina puede hallarse fuera del nido.", "Como medida preventiva se recomienda limpiar diariamente y quitar las fuentes de comida pegajosa, residuos dulces y grasosos y cerrar bien los artículos con fragancias.", "Como medida preventiva se recomienda identificar y sellar los sitios de ingreso. Las entradas de las hormigas obreras pueden ser bloqueadas mediante el uso de silicona, masilla, pegamento o yeso. Otra opción es colocar una barrera de vaselina o tiza ya que las hormigas no pueden caminar sobre estos productos.", "Como repelente se pueden usar condimentos como canela, menta, clavo de olor, ajo y pimienta negra, los cuales pueden ser diluidos en agua y rociados en muebles.", "Como repelentes se puede colocar hojas de laurel alrededor de la despensa de alimentos. Las hormigas también son repelidas por el vinagre, se recomienda mezclar partes iguales de vinagre y agua y luego rociarlo en áreas de almacenamiento y preparación de alimentos.", "Como repelente se recomienda colocar macetas con menta y clavo de olor en las ventanas o el jardín.", "Como medida preventiva se recomienda impedir el ingreso a fuentes de alimento de mascotas. Y si algún elemento está siendo invadido por hormigas, puede ser colocado sobre un plato con agua.", "Para el control de hormigas urbanas se recomienda rociar a cada individuo o a la boca del nido, con una solución de agua con vinagre en partes iguales.", "Un método de control es el uso de un cebo tóxico. Por ejemplo, mezclar agua con azúcar y ácido bórico o borato de sodio."],
    "oruga": ["El periodo crítico para el control es cuando las plantas aún son muy jóvenes o cuando generan brotes nuevos, ya que las plantas están más vulnerables.", "Un modo de controlar a las orugas y a los huevos, es mediante el uso de tierra Diatomea, la cual debe espolvorearse sobre las plantas regularmente. La oruga muere al ingerir y a la vez deshidrata a los huevos.", "Un modo de controlar a las orugas y a los huevos, es mediante el uso de una infusión de ajenjo. Se realiza el té y debe dejarse reposar por 24 horas. Luego se coloca el té en un aspersor y se rocía la planta a tratar.", "Una opción para disminuir el número de mariposas nocturnas, es colocar a la noche una luz con un panel de plástico donde las mariposas choquen al ser atraídas hacia la luz. Debajo del panel, debe colocarse una fuente con agua y detergente.", "Los enemigos naturales de las orugas en sus primeros estadios son las aves, avispas, moscas parasitoides, escarabajos, arañas u hormigas, que con su presencia controlan que las larvas o gusanos se conviertan en una plaga."],
    "pilme": ["Las prácticas de manejo y control se basan principalmente en realizar preparación del suelo y siembras tempranas, el uso de variedades de cultivos con defensas contra el escarabajo, el uso de repelente, y la aplicación de sustancias químicas activas.", "Una técnica para la prevención de pilmes se basa en una preparación temprana del suelo, de modo de exponer los estadios larval es a la acción de depredadores y a condiciones ambientales desfavorables para su desarrollo (luz, temperatura).", "Se recomienda para prevenir la presencia de pilmes, sembrar variedades de los cultivos susceptibles que posean defensas contra el escarabajo, como ser una fuerte barrera mecánica en base a tricomas ('pelos' que recubren la superficie de las hojas, pétalos o raíces).", "Para el control de pilmes se pueden utilizar algunos productos naturales como repelentes, por ejemplo en el sur de Chile se suelen colocar ramas del arbusto nativo conocido como 'duraznillo negro'."],
    "pulgon": ["Para prevenir el ataque por pulgones, se recomienda plantar cerca de las especies que se quieren proteger, plantas que funcionan como repelentes tales como la lavanda, la madreselva, el lupino, el ajo o la ortiga. Es importante asociar los cultivos con plantas aromáticas que liberan aromas que desorientan a los pulgones y no logran encontrar la especie que buscan.", "Para su detección se pueden utilizar trampas. Las trampas engomadas amarillas y las bandejas amarillas con agua son atrayentes de los pulgones alados.", "se utilizan plantas atrayentes como las flores que se asocian con los cultivos, tales como las caléndulas (calendula officinalis) con esto evitamos que los pulgones colonicen otras plantas.", "Un método de control orgánico muy efectivo es pulverizar las plantas afectadas con agua con jabón u otros preparados como alcohol de ajo, infusión de lavanda, tintura madre de copetes, macerado de ortiga, etc.", "Para favorecer el control biológico de los pulgones es necesario proteger a sus enemigos naturales, los sírfidos, las vaquitas de san antonio, las larvas de moscas y algunas avispas controlan a los pulgones alimentándose de ellos o parasitandolos", "Los enemigos naturale o insectos beneficos son atraídos por flores y plantas aromaticas, por ejemplo, el eneldo alberga vaquitas de san antonio", "Mantener suelos con materia orgánica (agregar compost, guano seco o lombricompuesto), favorecen el crecimiento de plantas sanas, las cuales no son tan atractivas para los pulgones, y tienen más capacidad en defenderse del ataque de insectos."],
    "mosca_blanca":["Evitar la alternancia de riegos, que el suelo no esté muy húmedo o muy seco. Mantener constante el riego. No usar fertilizantes químicos, usar compost o lombricompuesto. Se puede controlar con jabón potásico."],
    "sirfido":["Para atraerlos es importante asociar los cultivos con aromáticas y flores. Y dejar crecer plantas espontaneas como cobertura del suelo, mal llamadas malezas."],
    "vaquita": ["Para atraerlos es importante asociar los cultivos con aromáticas y flores. Y dejar crecer plantas espontaneas como cobertura del suelo, mal llamadas malezas."],
    "cochinilla": ["Asociar con aromáticas y flores, mantener el suelo fértil y la planta sana y vigorosa evitan el ataque de este insecto. El alcohol de ajo, la tierra de diatomeas, el aceite de neem pueden reducir las poblaciones, es importante controlar cuando las ninfas, el estadio mas sensible del insecto, avanzan por hojas y ramas para adherirse a las plantas y alimentarse."]
}

class ActionProvideSlottedInsectPreventInfo(Action):

    def name(self) -> Text:
        return "action_provide_slotted_insect_prevent_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        insect_name = unidecode(tracker.get_slot("insect_name").lower())
        if insect_name in insect_prevent_info_database:
            dispatcher.utter_message(
                text=choice(insect_prevent_info_database[insect_name]))
        else:
            dispatcher.utter_message(
                text=f"Todavía no tenemos información de prevención/atracción de {insect_name}")

        return []
    
# AID KIT ACTIONS -------------------------------------------------------------
aid_kit_product_prep_database = {
    "purinDeOrtiga": "Se utiliza la parte aérea de las plantas: 100 grs de ortiga x 5 ltrs de agua. Se deja fermentar durante 12 días.",
    "macDeOrtiga": "Se utiliza la parte aérea de las plantas: 1 kg x 10 litros de agua y macerar durante 12 horas.",
    "infDeManzanilla": "Se utilizan 50 gramos saproximadamente de flores secas o 100 g si son frescas. Se colocan las hojas frescas o secas de las plantas utilizadas en agua hirviendo, reposar tapado unos minutos. Utilizar tibia ",
    "infDeAjo": "Se machacan 75 grs. de ajo y se agregan a 10 litros de agua caliente. Aplicar frio",
    "macDeAji": "Macerar o machacar 500 gr. de ají seco, agregar 1 litro de agua y dejar reposar 24 hs., filtrar y mezclar con 20 litros de agua y 1 cucharada de jabón en pan.",
    "purinDeAjenjo": "Se utilizan las partes verdes y las flores. En fresco se utilizan 300 grs. x litro de agua y en planta seca 30grs. x litro de agua.",
    "infDeAjenjo": "Se prepara colocando las plantas frescas o secas con agua hirviendo, hasta lograr la infusión y posteriormente se deja enfriar y reposar durante 24 horas.",
    "decDeAjenjo": "Las partes verdes y flores se dejan en remojo durante 24 horas; luego se los hierve durante 20 minutos, se tapa y se deja enfriar.",
    "purDeCebYAjo": "Se utilizan bulbos y hojas a razón de 500 grs. x 10 litros de agua (si son plantas frescas) y 300 grs, x 10 litros de agua (si son plantas secas).",
    "decDeColaDeCab": "En 10 litros de agua se hierve 1 kg de cola de caballo fresca (o bien 150 grs. en polvo) durante 20 o 30 minutos. Luego de enfriado se coloca 1 por ciento de silicato sódico para aumentar la adherencia.",
    "alcoholDeAjo": "Extracto alcohólico: Para preparar 1 litro de extracto se deben moler 50 grs. de ajo y 50 grs. de ají picante. Macerarlos en 1 litro de alcohol 90° durante 7 días. Posteriormente filtrar y eliminar las partes gruesas del ajo y el ají. Puede conservarse hasta por 6 meses.",
    "infDeRudaYSalvia": "Se utilizan las partes verdes de ambas plantas. Se colocan en un tacho de agua unos 100 grs. de ruda y unos 100 grs. de salvia. Se completa con agua y se deja hasta que rompe el hervor. Se deja enfriar y posteriormente se filtra para eliminar el sobrante de hojas.",
    "purinDeTabaco": "Se utiliza el tabaco de 2 o más cigarrillos. Se los coloca en agua aproximadamente una 3 semanas. Posteriormente se filtra para eliminar los sólidos. Se mezcla con agua jabonosa de jabón blanco.",
    "infDeLavanda": "Se colocan ramas frescas o secas de lavanda en agua hirviendo. Se deja enfriar por 24 horas y se filtra para eliminar los elementos sólidos y se guarda en frasco.",
    "acEsenciales": "Hojas frescas.",
    "cenizas": "Se utilizan preferentemente cenizas de especies aromáticas. 3 a 4 kg de ceniza de acuerdo a kg leña). 10 litros de agua. ½ barra de jabón blanco o aceite agrícola.",
    "jabPotásico": "Se disuelve 25 gramos de jabón potasico en un litro de agua tibia. Para aplicarlo esta preparacion se diluye 200militros en un litro de agua, se puede sumar 250 ml de alcohol de ajo y una cucharada de tierra de diatomeas.",
    "diatomeas": "La tierra de diatomeas se extrae de  fósiles de rocas de algas fosilizadas, cubiertas por silicio extraída del fondo marino.",
    "caldoBordelés": "Ingredientes para 20 litros de agua: *200 grs de cal viva o apagada. 200 grs de sulfato de cobre. Materiales: 1 balde de plástico de 25 litros + 1 balde chico con capacidad para 2 litros + 1 paleta de madera para remover la mezcla + una varilla de hierro (o un machete) para probar la acidez de la mezcla. Preparación: 1)En un balde de plástico pequeño, colocar agua caliente y disolver los 200 grs de sulfato de cobre. 2) En otro balde más grande (mínimamente 25 litros) disolver la cal previamente apagada. 3) Una vez que tenemos los dos productos por separado, hacemos la mezcla vaciando el sulfato de cobre dentro del tacho de cal nunca al revés porque se corta. 4) Una vez mezclado y siguiendo con la remoción, colocar la varilla de hierro dentro de la mezcla para comprobar la acidez de la misma. Si la varilla se oxida, está muy ácida y por lo tanto hay que agregarle más cal para neutralizar. 5) Una vez que se llega a la acidez deseada (no oxida la varilla), el caldo está listo para usar."
}
aid_kit_product_usage_database = {
    "purinDeOrtiga": "Estos subproductos pueden aplicarse a las plantas durante todo el año. Normalmente se usa una dilución de 1:20, es decir 1 litro de purín x 20 litros de agua.",
    "macDeOrtiga":"Se aplica durante todo el año sobre troncos, ramas y ramitas. Se debe aplicar puro (sin diluir).",
    "infDeManzanilla":"La aplicación de estos productos sobre las plantas o en el suelo antes de sembrar en almacigos. Tiene propiedades antimicrobianas y fungicidas.",
    "infDeAjo":"El producto logrado se debe aplicar puro (sin diluir) sobre plantas y también sobre suelos. La época de aplicación es en primavera, 3 veces con intervalos de 3 días.",
    "macDeAji":"Se aplica sobre hojas y ramas de las plantas.",
    "purinDeAjenjo":"La aplicación es en primavera sobre las partes afectadas de las plantas. Se aplica el producto puro, sin diluir.",
    "infDeAjenjo":"La aplicación es en primavera sobre las partes afectadas de las plantas. Se aplica el producto puro, sin diluir.",
    "decDeAjenjo":"El producto se aplica como repelente cuando se observa las mariposas sobrevolando el cultivo de zanahorias.",
    "purDeCebYAjo":"En caso de ataque se aplica alrededor de los árboles en una dilución 1:10 (1 litro de producto x 10 litros de agua). En vuelo, se aplica sobre las plantas sin diluir.",
    "decDeColaDeCab":"Se aplica cuando aparecen los primeros síntomas de ataques de hongos en las plantas. Se debe aplicar en una dilución 1: 5, es decir 1 litro de producto por 5 litros de agua.",
    "alcoholDeAjo":"Se aplica cuando aparecen los primeros síntomas de ataques de hongos en las plantas. Se debe aplicar en una dilución 1: 5, es decir 1 litro de producto por 5 litros de agua.",
    "infDeRudaYSalvia":"Se aplica el líquido sobre las partes verdes de la planta sin diluir",
    "purinDeTabaco":"La mezcla se aplica con fumigadora manual sobre las hojas de las plantas. También se puede colocar el preparado de tabaco solo, al suelo.",
    "infDeLavanda":"Se realizan aplicaciones del producto puro sobre las plantas.",
    "acEsenciales":"Se utilizan las hojas, las cuales se entierran en los almácigos para que liberen sus sustancias activas",
    "cenizas": "El producto logrado se desparrama sobre los tablones de huerta en forma pareja En un balde de chapa de aproximadamente 20 litros hacer hervir 10 litros de agua y agregar la ceniza. El caldo debe hervir durante 30 minutos, removiéndose constantemente para obtener un buen producto. Posteriormente dejar enfriar, colar, envasar y guardar en lugar oscuro.",
    "lecheDescremada": "Se realiza aplicación sobre las partes afectadas de las plantas",
    "jabPotásico": "Este insecticida ecológico actúa por contacto sobre insectos de cuerpo blando que respiran por la piel. reblandece su protección superficial y les produce asfixia. No se aplican en horarios cuando hace mucho calor.",
    "diatomeas": "En seco: se espolvorea sobre la planta, es más efectivo. Diluido en agua una cucharada en un litro de agua.",
    "caldoBordelés": "Las dosis a aplicar varían de acuerdo al cultivo a tratar: * En viñedos se aplica dosis pura. * En cebolla, ajo y tomate la dosis es de 3:1, es decir: ¾ partes de caldo / ¼ litro de agua. * Para cultivos tales como arveja, habas, repollos, pepinos y zapallo, mezclar: 50 por ciento de caldo + 50 por ciento de agua. * Para cultivo de tomate y papa, cuando las plantas hayan alcanzado una altura de aproximadamente 30 cm, se debe utilizar: 2/3 partes de caldo por 1/3 de agua."
}
aid_kit_product_effect_database = {
    "purinDeOrtiga": "Es un estimulador del crecimiento de las plantas. Además previene la aparición de enfermedades en las plantas. También controla pulgones y ácaros.",
    "macDeOrtiga": "Protege contra el ataque del pulgón lanígero",
    "infDeManzanilla": "Protege a las semillas y plantas del ataque de hongos y de insectos chupadores tales como pulgones, trips.",
    "infDeAjo": "Inhibe la aparición de enfermedades causadas por hongos y es muy efectivo contra el ataque de pulgones y ácaros (arañuela roja).",
    "macDeAji": "El ají actúa por ingestión inhibiendo el apetito de los insectos. Ejerce acción insecticida, repelente, antiviral. Sus principios activos se concentran en la cáscara y en las semillas.",
    "purinDeAjenjo": "Es un producto recomendado como repelente contra la hormiga negra y los pulgones.",
    "infDeAjenjo": "Este producto está recomendado fundamentalmente para el control de ácaros (arañuela roja).",
    "decDeAjenjo": "Se recomienda el producto como repelente de la mosca de la zanahoria, es decir para evitar la postura de huevos.",
    "purDeCebYAjo": "Protege a las plantas de las enfermedades de hongos y repele insectos tales como la mosca de la zanahoria.",
    "decDeColaDeCab": "Recomendado en plantas con ataques de hongos, también se lo usa como repelente de muchos insectos y como controlador de insectos debido a su efecto adherente.",
    "alcoholDeAjo": "Este preparado está recomendado para el control de pulgones, ácaros, arañuela roja, gorgojos, moscas blancas, minador de las hojas y trips en cultivos hortícolas, florícolas.",
    "infDeRudaYSalvia": "Este producto está recomendado para el control de pulgones, cochinillas y moscas blancas.",
    "purinDeTabaco": "Se recomienda su uso para el control de pulgones, ácaros y orugas",
    "infDeLavanda": "Es un repelente de todo tipo de insectos. También se ha observado que posee efectos sobre hongos.",
    "acEsenciales": "Planta que tiene propiedades repelentes, insecticidas, acaricidas (arañuela) e inhibidora de crecimiento por lo que controla áfidos (pulgones), polillas (orugas), arañas rojas y moscas.",
    "cenizas": "Se utiliza el producto como repelente de orugas, chinches y pulgones. La dosis de aplica- ción es de 2 a 4 litros de caldo / 20 litros de agua. Como adherente se puede añadir 2 a 3 cucharadas de aceite agrícola o ½ barra de jabón disuelto. Controla enfermeda- des como: viruela, cochinillas cerosas, botritis y cenicilla.",
    "lecheDescremada": "Produce efectos sobre los pulgones y también sobre hongos de las plantas.",
    "jabPotásico": "Es efectivo para eliminar y combatir plagas como pulgón, la mosca blanca, cochinillas, trips y araña roja.",
    "diatomeas": "Se la utiliza como preventivo, y en lugares donde la plaga ya esta presente, para bajar la carga poblacional de ésta. Muy buena eficacia, actuando para insectos masticadores y que se arrastran, como orugas, babosas y caracoles, para tucuras, hormigas, colocar directo en el hormiguero o en el camino que recorren.",
    "caldoBordelés": "* Enfermedades causadas por hongos, tizones y mildiu. * No aplicar sobre árboles frutales cuando los mismos tienen sus hojas desplegadas. * Es recomendable utilizar este producto inmediatamente después de prepararlo. * En su preparación siempre deben ser utilizados envases plásticos."
}

aid_kit_commercial_product_database = {
    "jabPot_com": "El jabón potásico es un compuesto formado por agua, aceite vegetal e hidróxido de potasio. El jabón potásico es efectivo para eliminar y combatir plagas como el pulgón, la mosca blanca, la cochinilla, los trips o la araña roja.",
    "diatomeas_com": "La tierra de diatomeas se extrae de  fósiles de rocas de algas fosilizadas, cubiertas por silicio extraída del fondo marino. Se la utiliza como preventivo, y en lugares donde la plaga ya esta presente, para bajar la carga poblacional de ésta. Muy buena eficacia, actuando para insectos masticadores y que se arrastran, como orugas, babosas y caracoles, para tucuras, hormigas, colocar directo en el hormiguero o en el camino que recorren. Se esparce una cantidad de producto sobre las partes del cultivo, y alrededor del mismo, o se diluye en agua",
    "acDeNeem": "Se extrae de una planta de la misma familia que el paraiso hay que tener en cuenta que el Neem no mata a las plagas de forma instantánea como los pesticidas sintéticos, sino que las va inhibiendo la alimentación, reproducción y crecimiento de los insectos. Su acción es relativamente lenta y tarda en hacer efecto entre 5 y 7 días",
    "acMineral": "Se usa de manera preventiva para el control de insectos y ácaros que atacan las plantas. Como cochinillas, pulgones, moscas blancas.También ejerce un buen efecto de control sobre ácaros (arañas rojas y amarillas)."
}

class ActionProvideAidKitProducts(Action):

    def name(self) -> Text:
        return "action_provide_aid_kit_products"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        aid_kit_product_name = tracker.get_slot("product_name")

        if aid_kit_product_name in aid_kit_product_prep_database:
            dispatcher.utter_message(
                text=f"Preparación:\n{aid_kit_product_prep_database[aid_kit_product_name]}")
       
        if aid_kit_product_name in aid_kit_product_usage_database:
            dispatcher.utter_message(
                text=f"Utilización:\n{aid_kit_product_usage_database[aid_kit_product_name]}")
        
        if aid_kit_product_name in aid_kit_product_effect_database:
            dispatcher.utter_message(
                text=f"Efecto:\n{aid_kit_product_effect_database[aid_kit_product_name]}")
            
        if aid_kit_product_name in aid_kit_commercial_product_database:
            dispatcher.utter_message(
                text=f"{aid_kit_commercial_product_database[aid_kit_product_name]}")
        return []

# WASTE ACTIONS ---------------------------------------------------------------

translate_waste_eng_to_spa = {
    "glass": "vidrio",
    "cardboard": "cartón",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "trash": "otros residuos"
}

class ActionIdentifyWaste(Action):

    def name(self) -> Text:
        return "action_identify_waste"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        learn = load_learner("./actions/", 'residuos_fastaiV1.pkl')

        img_url = tracker.latest_message["text"]
        img_unique_id = tracker.latest_message["metadata"]["message"]["photo"][-1]["file_unique_id"]

        img_filename = f"./actions/photos/identify_waste/{img_unique_id}.jpg"

        urlretrieve(img_url, img_filename)

        with open(img_filename, "rb") as image:
            img_bytes = image.read()
            img = open_image(BytesIO(img_bytes))
            prediction = learn.predict(img)
            prediction_str = translate_waste_eng_to_spa[str(prediction[0])]
            confidence_top_prediction = max([round(p,3) for p in prediction[2].numpy()])
            dispatcher.utter_message(text=f"Identifico que es {prediction_str} con una confianza del {round(confidence_top_prediction*100, 4)}%")

        return []


class ActionProvideCompostingGeneralInfo(Action):

    def name(self) -> Text:
        return "action_provide_composting_general_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # dispatcher.utter_message(json_message={
        #     "text": '*bold text* \n_italic text_\n [text](https://www.google.com.ar/)\n `inline fixed-width code`\n ```pre-formatted fixed-width code block```',
        #     "parse_mode": 'MarkdownV2'
        # })

        dispatcher.utter_message(
            image="https://imgur.com/OvUfeNI.png")
        return []
    
class ActionProvideCompostProblems(Action):

    def name(self) -> Text:
        return "action_provide_compost_problems"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # dispatcher.utter_message(json_message={
        #     "text": '*bold text* \n_italic text_\n [text](https://www.google.com.ar/)\n `inline fixed-width code`\n ```pre-formatted fixed-width code block```',
        #     "parse_mode": 'MarkdownV2'
        # })

        dispatcher.utter_message(
            image="https://imgur.com/AAod9fT.png")
        return []



