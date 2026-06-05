import requests
import os 
from pathlib import Path
import yaml
import time
from bs4 import BeautifulSoup

characters_yaml = 'characters.yaml'
raw_dir = Path('data/character_pages')
raw_dir.mkdir(parents=True, exist_ok=True)


headers={"User-Agent": "Mozilla/5.0"}

def fetch_pages():
    """
    Fetches a selection of One Piece Character Wikipedia Pages
    """
    with open(characters_yaml, 'r') as f:
        pages = yaml.safe_load(f)["pages"]

    for page in pages:
        character = page["name"]
        url = page["url"]

        filename = character.replace(' ', '_')
        filepath = raw_dir / f'{filename}.txt'

        print(f'Working on {character}...')

        try:
            response = requests.get(url, timeout=30, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", id="mw-content-text")
            elements = content.find_all(["p", "h1", "h2", "h3", "h4", "li"])

            cleaned_elements = []

            for el in elements:
                if el.name in ['h2', 'h3'] and el.get_text(strip=True).lower() in ['see also', 'references', 'notes']:
                    break
                
                txt = el.get_text(' ', strip=True)

                if el.name in ['h1', 'h2']:
                    cleaned_elements.append(f'\n\n ===== {txt} =====\n\n')
                elif el.name in ['h3', 'h4']:
                    cleaned_elements.append(f'\n == {txt} ==\n')
                elif el.name == 'p':
                    cleaned_elements.append(f'{txt}\n')
                elif el.name == 'li':
                    cleaned_elements.append(f'- {txt}')
                else:
                    cleaned_elements.append(txt)


            text = "\n".join(cleaned_elements)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f'Wrote {character}')

        except Exception as e: 
            print(f'Error fetching {character}: {e}')
        
        time.sleep(1)