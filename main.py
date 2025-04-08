from generator.Generator_Main import train_generator
from UI.Dungeon_UI import launch_ui
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "train"], default="ui")
    args = parser.parse_args()

    if args.mode == "ui":
        launch_ui()
    elif args.mode == "train":
        train_generator(num_episodes=1000)

if __name__ == "__main__":
    main()
