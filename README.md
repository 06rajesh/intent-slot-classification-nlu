# Natural Language Understanding: Intent Slot Classification

Extracting intents and slots from textual command using BERT `sequence output` and 
`pooled output`. Trained a single model for both tasks.

### Dependencies
* torch>=1.8.0
* scipy
* tqdm
* transformers
* unidecode
* tensorboard

### Sample Output
```buildoutcfg
    "0": {
        "intent": "BookRestaurant",
        "text": "I'm looking for a local cafeteria that has wifi accesss for a party of 4",
        "slots": {
            "spatial_relation": "local",
            "restaurant_type": "cafeteria",
            "facility": "wifi",
            "party_size_number": "4"
        }
    },
    "1": {
        "intent": "AddToPlaylist",
        "text": "Add As I Was Going to St Ives to the fantasia playlist.",
        "slots": {
            "entity_name": "as i was going to st ives",
            "playlist": "fantasia"
        }
    },
    "2": {
        "intent": "BookRestaurant",
        "text": "book for one in Indiana at a restaurant",
        "slots": {
            "party_size_number": "one",
            "state": "indiana",
            "restaurant_type": "restaurant"
        }
    },
    "3": {
        "intent": "AddToPlaylist",
        "text": "put this album on my conexiones list",
        "slots": {
            "music_item": "album",
            "playlist_owner": "my",
            "playlist": "conexiones"
        }
    },
```




