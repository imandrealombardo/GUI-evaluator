import json
import os
import random
from tkinter import Toplevel, Button, messagebox, Tk, Label, ttk
from PIL import ImageTk, Image
from pynput.keyboard import Listener, Key
import numpy as np
import cv2
import pycocotools.mask as mask_util
from collections import defaultdict
import csv
import sys
import pandas as pd
import argparse

key_pressed = None
valid_key_pressed = False

def reverse_map_faces(face):
    # Function that maps the face name to the corresponding index
    # Mapping: 'front' -> 0, 'right' -> 1, 'back' -> 2, 'left' -> 3, 'top' -> 4, 'bottom' -> 5
    face_mapping = {'front': 0, 'right': 1, 'back': 2, 'left': 3, 'top': 4, 'bottom': 5}
    return face_mapping.get(face)

def are_there_panoramas_left(image_dir, csv_path):
    panorama_dirs = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    if os.path.exists(csv_path):
        annotated_panoramas = set(pd.read_csv(csv_path)['pano_id'].values)
        panorama_dirs = [d for d in panorama_dirs if os.path.basename(d) not in annotated_panoramas]
    return panorama_dirs != []

def wait_for_window(root):
    root.wait_visibility()  # Wait until the window becomes visible
    while root.winfo_width() <= 1 or root.winfo_height() <= 1:
        root.update()  # Let Tkinter process events, this will update window size if it's been resized

def split_filename(f):
    # Split the string into the panorama id with underscore and the rest
    pano_face_extension = f.rsplit('_', 1)
    
    try:
        # Further split the second part to get the face and the extension
        face_extension = pano_face_extension[1].rsplit('.', 1)
    except:
        print(f"Error splitting filename {f}")
        return None, None
    
    # Return the results
    return pano_face_extension[0], face_extension[0]

def resize_image(img, max_width, max_height):
    # Get the current size of the image
    original_width, original_height = img.size

    # Calculate the ratio of the old width and height to the new ones
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # Choose the smaller ratio
    new_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height of the image
    new_width = int(original_width * new_ratio)
    new_height = int(original_height * new_ratio)

    # Resize the image
    img = img.resize((new_width, new_height))

    # Return the resized image
    return img

# Popup a window with three buttons
def popup_message_with_buttons(message):    
    # Create a new popup window
    top = Toplevel()
    top.title("Start")
    Label(top, text=message).pack(padx=10, pady=10)
    choice = None

    def on_button_click(button_choice):
        nonlocal choice
        choice = button_choice
        top.destroy()

    # Define a buttons dictionary with text and choice
    buttons = {'Task 1: Is the object mask located on a sidewalk? [yes/no] [1/2]': 1, 
               'Task 2: Does the object mask represent an obstacle on the sidewalk? [yes/no] [1/2]': 2, 
               'Task 3: Rate the object mask quality [fail/pass/excellent] [1/2/3]': 3}
    for text, choice in buttons.items():
        Button(top, text=text, command=lambda choice=choice: on_button_click(choice)).pack(padx=10, pady=10)

    # Center the popup window
    top.update_idletasks()
    width = top.winfo_width()
    height = top.winfo_height()
    x = (top.winfo_screenwidth() // 2) - (width // 2)
    y = (top.winfo_screenheight() // 2) - (height // 2)
    top.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    top.grab_set()  # Grab the focus of the window
    top.wait_window(top)  # Wait for the window to be closed

    return choice


# Load JSON files from a directory
def load_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(directory, filename)) as f:
                    json_files.append(json.load(f))
            except Exception as e:
                print(f"Error loading json file {filename}: {e}")
    return json_files

# Load images from a directory
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            try:
                image = Image.open(os.path.join(directory, filename))
                images.append(ImageTk.PhotoImage(image))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images

def wait_for_key(e, root, task):
    global key_pressed  # We are using the global variable key_pressed
    global valid_key_pressed  # We are using the global variable valid_key_pressed

    if e == Key.esc:  # If escape key is pressed
        print('Escape pressed (key)')
        root.destroy()
        return False

    try:
        if hasattr(e, 'char'):  # The event is a KeyCode
            key_pressed = e.char

        elif hasattr(e, 'vk'):  # The event is a Key
            key_pressed = e.vk

    except AttributeError:
        print('AttributeError')
        return True  # Keep listening

    if (task == 3 and key_pressed in ['1', '2', '3']) or \
       (task != 3 and key_pressed in ['1', '2']):
        valid_key_pressed = True  # Set valid_key_pressed to True when a valid key is pressed
        return False  # Stop the listener

    return True  # Keep listening if invalid key is pressed

def show_images(root, model_dirs, image_dir, model_masks, task, model_dir_mapping, window_title, csv_path):
    root.title(window_title)  # Set the window title

    # Make the grid cells expand to fill the available space
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=1)

    # Define a Label to show the count of panoramas and models
    counter_label = Label(root, text="", anchor="center", justify="left")
    #counter_label = Label(root, text="", anchor="center", justify="center")
    counter_label.grid(row=0, column=0, sticky="nsew")

    # Create a progress bar
    progress = ttk.Progressbar(root, length=500, mode='determinate')
    progress.grid(row=1, column=0, sticky="nsew")

    # Keep a list of references to image objects to prevent them from being garbage collected
    image_references = []
    
    # Track user choices for each mask
    user_choices = defaultdict(lambda: defaultdict(list))  
    
    panorama_dirs = [os.path.join(image_dir, d) for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    # Print panoramas directories
    #print(f'Panorama directories: {panorama_dirs}')

    # Filter already annotated panoramas by checking if the panorama directory name
    # is in the .csv file. The name will be the same as the pano_id cell.
    if os.path.exists(csv_path):
        annotated_panoramas = set(pd.read_csv(csv_path)['pano_id'].values)
        panorama_dirs = [d for d in panorama_dirs if os.path.basename(d) not in annotated_panoramas]

    #print(f'Panorama directories after filtering: {panorama_dirs}')

    # Count the total number of images (panoramas * faces)
    total_images = sum([len(os.listdir(d)) for d in panorama_dirs])

    half_images = total_images // 2  # Use integer division to get the midpoint
    #print(f"Total images: {total_images}")1

    image_counter = 0  # Initialize counter for the number of images show

    for panorama_idx, panorama_dir in enumerate(panorama_dirs):

        print(f'Analyzing {panorama_dir}')

        # Get all the image names in the panorama directory
        image_names = [f for f in os.listdir(panorama_dir) if f.endswith('.png')]
        
        for image_name in image_names:
            #print(f'Analyzing image {image_name}')
            pano_id, face_name = split_filename(image_name)
            face_idx = reverse_map_faces(face_name)

            #print('Pano_id: ', pano_id)

            #print(f'Analyzing image {image_name} with pano_id {pano_id} and face_idx {face_idx}')

            # Load the image
            img = Image.open(os.path.join(panorama_dir, image_name))
            
            # Randomize the order of the model directories for each image
            random.shuffle(model_dirs)
            
            # For each of the model directories...
            for model_idx, model_dir in enumerate(model_dirs):
                # Update the model counter
                counter_label['text'] = f"Panorama {panorama_idx + 1}/{len(panorama_dirs)}\n" \
                                        f"Face: {face_name}\nModel {model_idx + 1}"

                #print(f"We entered the model dir: {model_dir}")
                # Get the masks for the current pano_id and face_idx
                masks = model_masks[model_dir].get(pano_id, {}).get(face_idx, [])
                # Check that the face has masks
                if len(masks) == 0:
                    print(f'Face {face_idx} has no masks. Skipping...')

                    continue

                # For each mask, overlay it on the image
                for mask_dict in masks:
                    # Decode the mask
                    rle = mask_dict['segmentation']
                    mask = mask_util.decode(rle)

                    # Create an output mask and apply the decoded mask to it
                    output_mask = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
                    output_mask[mask == 1] = [255, 255, 255]  # White mask

                    # Overlay the output mask on the image with a 0.3 alpha value
                    overlay_img = Image.fromarray(cv2.addWeighted(np.array(img), 0.7, output_mask, 0.6, 0))

                    # Resize the image to fit the window
                    #overlay_img = resize_image(overlay_img, root.winfo_width(), root.winfo_height())
                    root.after(100, lambda: resize_image(overlay_img, root.winfo_width(), root.winfo_height()))

                    # Convert the processed image to a PhotoImage and display it
                    image = ImageTk.PhotoImage(overlay_img)
                    image_references.append(image)  # Keep a reference to the image object
                    label = Label(root, image=image)
                    #label.pack()
                    label.grid(row=2, column=0, sticky="nsew")
                    
                    root.update()  # Update the window to make the image visible

                    global valid_key_pressed
                    global key_pressed
                    print(f'valid_key_pressed before entering the Listener while True loop: {valid_key_pressed}')

                    while True:  # Wait until a valid key is pressed
                        with Listener(on_press=lambda e: wait_for_key(e, root, task), suppress=True) as listener:  # Wait for a key press
                            listener.join()  # Wait for the listener to stop

                        if valid_key_pressed:  # Exit the loop if a valid key was pressed
                            valid_key_pressed = False  # Reset valid_key_pressed to False for the next image
                            break
                    user_choices[pano_id][face_idx].append((key_pressed, model_dir_mapping[model_dir]))  # Record the user's choice
                    key_pressed = None # Reset key_pressed for the next image
                    label.pack_forget()  # Hide the current image

            # Update the progress bar after each image
            image_counter += 1  # Increment counter after each image
            progress['value'] = image_counter / total_images * 100
            root.update_idletasks()
            if image_counter == half_images:
                messagebox.showinfo("Notification", "Great job, you are halfway through! Please click OK to continue.")

        # Write the user's choices to a .csv file after each panorama_dir
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(["pano_id", "face_idx", "choices", "model"])  # Write header
            for pano_id, face_choices in user_choices.items():
                for face_idx, model_choices in face_choices.items():
                    # Group choices by model
                    choices_by_model = defaultdict(list)
                    for choice, model in model_choices:
                        choices_by_model[model].append(choice)
                    # Write grouped choices to .csv
                    for model, choices in choices_by_model.items():
                        writer.writerow([pano_id, face_idx, ','.join(choices), model])

    # Clear the choices for the current panorama_dir
    user_choices.clear()

    # After all images have been displayed, show a final pop-up message
    counter_label['text'] = ""  # Clear the label
    messagebox.showinfo("Notification", "Task completed. Thanks for your time :)")
    progress.destroy()  # Destroy the progress bar

    root.destroy()  # Close the script when OK is pressed

def main(args):
    root = Tk()
    root.withdraw()  # Hide the main window when the script starts

    global task  # Make the task variable global to access it in the wait_for_key function
    task = popup_message_with_buttons("Welcome! Please select a task. \n Press esc to quit at any moment.")  # Show a pop-up message
    print(f"User selected: Task {task}.")

    task_mapping = {'Task 1: Is the object mask located on a sidewalk? [yes/no] [1/2]': 1, 
               'Task 2: Does the object mask represent an obstacle on the sidewalk? [yes/no] [1/2]': 2, 
               'Task 3: Rate the object mask quality [fail/pass/excellent] [1/2/3]': 3}
    
    # Find the key in task_mapping that corresponds to the value of chosen_task
    task_text = [k for k, v in task_mapping.items() if v == task][0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir
    # Get a list of all model directories, skip any non-directory files
    model1 = os.path.join(script_dir, data_dir, args.model1)
    model2 = os.path.join(script_dir, data_dir, args.model2)
    model3 = os.path.join(script_dir, data_dir, args.model3)
    model_dirs = [model1, model2, model3]
    image_dir = os.path.join(script_dir, data_dir, args.images_dir)

    # Create a set of all panorama ids, filter out non-directory files
    #panorama_ids = {os.path.splitext(f)[0] for d in os.listdir(image_dir) if os.path.isdir(os.path.join(script_dir, d)) for f in os.listdir(os.path.join(image_dir, d)) if f.endswith('.png')}
    panorama_ids = {d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))}
    #print(f"Panorama ids: {panorama_ids}")

    # Create a dictionary that maps each model directory to its respective grouped_panos dictionary
    model_masks = {}
    for model_dir in model_dirs:
        json_file = os.path.join(model_dir, args.masks_json)
        with open(json_file) as f:
            panos_info = json.load(f)
        
        grouped_panos = {}
        for pano in panos_info:
            pano_id = pano['pano_id']
            face_idx = pano['face_idx']
            
            if pano_id not in panorama_ids:  # Skip pano_ids that are not in panorama_dirs
                continue

            if pano_id not in grouped_panos:
                grouped_panos[pano_id] = {}

            if face_idx not in grouped_panos[pano_id]:
                grouped_panos[pano_id][face_idx] = []

            grouped_panos[pano_id][face_idx].append(pano)

        model_masks[model_dir] = grouped_panos

    root.deiconify()  # Show the main window again
    if args.os == 'macOS':
        root.attributes('-fullscreen', True)  # Set the root to fullscreen
    else:
        root.attributes('-zoomed', True) # Set the root to fullscreen

    model_dir_mapping = {}
    for model_dir in model_dirs:
        if "baseline" in model_dir:
            model_dir_mapping[model_dir] = 0
        elif "algorithm1" in model_dir:
            model_dir_mapping[model_dir] = 1
        elif "algorithm2" in model_dir:
            model_dir_mapping[model_dir] = 2
        else:
            model_dir_mapping[model_dir] = 3  # Or any other default value you'd like

    # If it's the first time running the script for a task, create a new .csv file
    csv_path = os.path.join(data_dir, f'task_{task}_results.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pano_id", "face_idx", "choices", "model"]) # Write header

    '''# Create a .csv file called no_masks.csv to store the pano_ids that don't have any masks
    if not os.path.exists('no_masks.csv'):
        with open('no_masks.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pano_id", "face_idx"])'''

    if not are_there_panoramas_left(image_dir, csv_path):
        print('No panoramas left to annotate!')
        root.destroy()
    else:
        wait_for_window(root)  # Wait for the window to be ready
        show_images(root, model_dirs, image_dir, model_masks, task, model_dir_mapping, task_text, csv_path)

    root.mainloop()

# Run the program
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--images_dir', type=str, default='dataset', help='images directory')
    parser.add_argument('--masks_json', type=str, default='input_coco_format_100.json', help='masks json file')
    parser.add_argument('--model1', type=str, default='baseline', help='first model to evaluate')
    parser.add_argument('--model2', type=str, default='algorithm1', help='second model to evaluate')
    parser.add_argument('--model3', type=str, default='algorithm2', help='third model to evaluate')
    parser.add_argument('--os', type=str, default='macOS', help='OS used for the evaluation')

    args = parser.parse_args()

    main(args)