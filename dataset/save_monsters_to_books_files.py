import json
import os

from dataset.constants import ORIGINAL_DATA_FOLDER, SAVE_DIR


def get_json_file_name(name: str) -> str:
    return os.path.join(SAVE_DIR, f"{os.path.basename(name)}.jsonl")


def save_monsters_from_dir(dir_path: str, name_of_the_file: str) -> tuple[bool, int]:
    no_npcs_found = True
    monsters_loaded = 0
    subfolders = [f.path for f in os.scandir(dir_path) if f.is_dir()]

    for folder in subfolders:
        no_npcs_found, monsters_from_subfolder = save_monsters_from_dir(
            folder, name_of_the_file
        )
        monsters_loaded += monsters_from_subfolder

    files_list = [
        file
        for file in os.listdir(dir_path)
        if file.endswith(".json") and not file.startswith("_")
    ]
    filename = get_json_file_name(name_of_the_file)

    with open(filename, "a+") as bestiary_file:
        for monster in files_list:
            with open(
                os.path.join(dir_path, monster), encoding="utf-8"
            ) as monster_file:
                monster_json = json.load(monster_file)
                if monster_json.get("type", "").lower() == "npc":
                    json.dump(monster_json, bestiary_file)
                    bestiary_file.write("\n")
                    no_npcs_found = False
                    monsters_loaded += 1

    return no_npcs_found, monsters_loaded


def clean_files(filenames: list[str]):
    for name_of_the_file in filenames:
        filename = get_json_file_name(name_of_the_file)
        if not os.path.exists(filename):
            continue
        with open(filename, "w"):
            pass


def save_main_folders(data_folder: str):
    main_subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    clean_files(main_subfolders)

    all_folders_number = 0
    skipped_folders = 0
    monsters_loaded = 0

    for folder in main_subfolders:
        filename = os.path.basename(folder)
        print(f"Loading: {filename}")
        no_npcs_found, subfolder_monsters_loaded = save_monsters_from_dir(
            folder, filename
        )
        monsters_loaded += subfolder_monsters_loaded

        if no_npcs_found:
            print(f"Remove {filename} (no monsters found)")
            os.remove(get_json_file_name(folder))
            skipped_folders += 1
        else:
            all_folders_number += 1

    print("Summary")
    print(
        f"""
        Files created: {all_folders_number}
        Subfolders without monsters: {skipped_folders}
        All found monsters jsons: {monsters_loaded}
        """
    )


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_main_folders(ORIGINAL_DATA_FOLDER)
