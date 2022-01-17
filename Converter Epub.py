import epub_conversion
import xml_cleaner
import re
from pathlib import Path
from epub_conversion.utils import open_book, convert_epub_to_lines



def get_text_from_epub(epub_file_path: Path, output_dir_path='./') -> str:

    book = open_book(str(epub_file_path))
    lines = convert_epub_to_lines(book)
    cleaner = re.compile('<.*?>')

    output_txt_filename = f'{output_dir_path}/{epub_file_path.stem}.txt'
    with open(output_txt_filename, 'w', encoding='utf-8') as txt_file:
        for line in lines:
            line = re.sub(cleaner, '', line)
            txt_file.write(line)

    return output_txt_filename


def list_directory(source_dir_path_name: str, target_dir_path_name: str) -> list:
    """
    source_dir_path_name: путь до папки, где лежат книжки
    target_dir_path_name: путь до папки, шде будут лежать .txt файлы
    """
    path = Path(source_dir_path_name)
    for ebook in path.iterdir():
        txt_file_name = get_text_from_epub(ebook, target_dir_path_name)

list_directory("./Books/Detectives/","./Books/DetectivesTXT/")
list_directory("./Books/Space Science/","./Books/Space ScienceTXT/")