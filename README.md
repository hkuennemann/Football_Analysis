# Football_Analysis
In this project I utilized computer vision to detect players, the ball and the referees of a short clip of a football match. Additionally to detecting, I also implemented a logic that calculates the distance and speed each player ran, relative to the camera movement.

![Readme_pic](output_videos/output_image.png)

## Libaries

- Camera Movement Estimator: Getting the camera movement, adjusting the positions to tracks and drawing the camera movement (Box in the upper left corner of the result).
- Player Ball Assigner: Library to track which player has the ball, i.e. ball posession.
- Speed and Distance Estimator: Calculating the speed and distance of players taking the camera movement into account. The results are then drawn underneath the players.
- Team Assigner: Assigns each player to a team. The team color is derived thorugh the kmeans clustering algorithm.
- Trackers: Adds the position to each players tracks, interpolates ball position for frames in which the ball was not found by the computer visoin algorithm and draws various annotations and shapes on the output frames.
- View Transformer: Corrects the perspective distortion introduced by angled camera views, allowing for accurate mapping of pixels to real-world distances on the football field.
- Utils: Utility library for simple distance and tracking-box calculations and for reading and saving a video.

## Used Models, Algorithms, and Concepts
- YOLO: AI-based object detection model
- K-means: Pixel segmentation and clustering for detecting t-shirt colors
- Optical Flow: Measures camera movement
- Perspective Transformation: Models scene depth and perspective

## Project Overview
This project is  based on the work of Abdullah Tarek, who shared a comprehensive approach to football analysis through his [GitHub repository](https://github.com/abdullahtarek/football_analysis) and [YouTube video](https://www.youtube.com/watch?v=neBZ6huolkg&list=LL). 