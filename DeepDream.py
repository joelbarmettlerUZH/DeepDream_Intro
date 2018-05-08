import PIL.Image
from deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import cv2
import os
import shutil

class DeepDream:

    def __init__(self, name, image):
        self.__name = name
        self.__image = image
        if not os.path.exists("./dream/" + self.__name):
            os.makedirs("./dream/" + self.__name)

    def singleDream(self, output, layer=3):
        # Chose Layer to enhance
        layer_tensor = model.layer_tensors[layer]
        file_name = self.__image
        img_result = load_image(filename='{}'.format(file_name))

        # Call recursive optimization to generate dream
        img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_result,
                                        # how clear is the dream vs original image
                                        num_iterations=20, step_size=1.0, rescale_factor=0.5,
                                        # How many "passes" over the data. More passes, the more granular the gradients will be.
                                        num_repeats=8, blend=0.2)
        # Make sure output results are valid pixel values
        img_result = np.clip(img_result, 0.0, 255.0)
        img_result = img_result.astype(np.uint8)
        # Generate and save image out of numpy array
        result = PIL.Image.fromarray(img_result, mode='RGB')
        result.save("./dream/" + self.__name + "/" + output)


    def startDreaming(self):
        # Define dream name
        dream_name = self.__name
        # Create output folder for dream frames if needed
        if not os.path.exists("./dream/" + dream_name):
            os.makedirs("./dream/" + dream_name)
        # Copy initial image as the first Frame
        shutil.copyfile(self.__image, "./dream/" + dream_name + "/img_0.jpg")
        # Open first frame and get its size
        im = PIL.Image.open(self.__image)
        x_size, y_size = im.size
        # Dream for at most max_count frames
        created_count = 0
        max_count = 30 * 60 * 15    # Max of 15 min at 30 FPS
        for i in range(0, 9999999999999999):
            # Search for already existing frames
            if os.path.isfile('dream/{}/img_{}.jpg'.format(dream_name, i + 1)):
                print('{} already exists, continuing along...'.format(i + 1))

            else:
                # Define layer boundaries
                starting_layer = 1
                max_layer = 11
                change_after = 30
                # Calculate current layer - auto increase layers over time / over i
                layer = min(int((1 / change_after) * i) + starting_layer, max_layer)
                print("Enhancing layer {}".format(layer))
                layer_tensor = model.layer_tensors[layer]
                # Loading image
                img_result = load_image(filename='dream/{}/img_{}.jpg'.format(dream_name, i))

                # Define zoom speed in pixels
                x_trim = 2
                y_trim = 1
                # Zoom image
                img_result = img_result[0 + x_trim:y_size - y_trim, 0 + y_trim:x_size - x_trim]
                img_result = cv2.resize(img_result, (x_size, y_size))

                # Use these to modify the general colors and brightness of results.
                # results tend to get dimmer or brighter over time, so you want to
                # manually adjust this over time.

                # +2 is slowly dimmer
                # +3 is slowly brighter
                img_result[:, :, 0] += 2  # reds
                img_result[:, :, 1] += 2  # greens
                img_result[:, :, 2] += 2  # blues

                # Again make sure to keep pixel values in range
                img_result = np.clip(img_result, 0.0, 255.0)
                img_result = img_result.astype(np.uint8)

                # Recursively optimize / dream image with 2 repeats per frame
                img_result = recursive_optimize(layer_tensor=layer_tensor,
                                                image=img_result,
                                                num_iterations=2,
                                                step_size=1.0,
                                                rescale_factor=0.5,
                                                num_repeats=1,
                                                blend=0.2)

                # Clip again
                img_result = np.clip(img_result, 0.0, 255.0)
                img_result = img_result.astype(np.uint8)

                # Performing histogram equalizaiton
                img_yuv = cv2.cvtColor(img_result, cv2.COLOR_BGR2YUV)

                # equalize the histogram of the Y channel
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

                # convert the YUV image back to RGB format
                equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                img_result = cv2.addWeighted(img_result, 0.98, equalized_image, 0.02, 0)
                # Sharpen n times
                for s in range(0):
                    blurred = cv2.GaussianBlur(img_result, (3, 3), 0)
                    img_result = cv2.addWeighted(img_result, 1.5, blurred, -0.5, 0)

                # Generate image from numpy array and save it into dream folder
                result = PIL.Image.fromarray(img_result, mode='RGB')
                result.save('dream/{}/img_{}.jpg'.format(dream_name, i + 1))

                created_count += 1
                if created_count > max_count:
                    break

    def toVideo(self):
        dream_name = self.__name
        dream_path = "dream/{}".format(dream_name)
        # Open first Frame and get size of video
        im = PIL.Image.open(dream_path + "/img_0.jpg")
        x_size, y_size = im.size

        # Create openCV video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        out = cv2.VideoWriter(dream_path + '/{}.mp4'.format(dream_name), fourcc, 20, (x_size, y_size))

        # For at most the length of a video, loop over frames till last one is found
        for i in range(30 * 60 * 15):
            if os.path.isfile('dream/{}/img_{}.jpg'.format(dream_name, i + 1)):
                print('{} already exists, continuing along...'.format(i + 1))
            else:
                dream_length = i
                break

        # Loop over all the frames and add them to video writer
        for i in range(dream_length):
            img_path = os.path.join(dream_path, "img_{}.jpg".format(i))
            print(img_path)
            frame = cv2.imread(img_path)
            out.write(frame)

        # Save video
        out.release()

if __name__ == "__main__":
    dreams = ["noise_XXL", "galaxy", "woman"]
    for d in dreams:
        dream = DeepDream(d, "{}.jpg".format(d))
        for i in range(12):
            print("Processing layer {}".format(i))
            dream.singleDream("noise_layer{}.jpg".format(i), layer=i)
    # dream.startDreaming()
    # dream.toVideo()