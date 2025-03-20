# Volley vision

An ongoing project focused on object detection in order to analyze and track elements in a volleyball game. I'm still exploring the project's direction, but the initial idea was to develop an app or website capable of processing video footage in order to provide statistics, player performance analysis, and match insights.

Current progress:

![volley_gif](https://github.com/user-attachments/assets/d78101e1-6512-4e59-92d5-2a90488f1924)

### Tracking:
- Players (working quite well)
- Ball (still needs some improvement)
- Referee (not bad)
- Actions (poor)
- Court :construction: (in construction)

### Models:
All models are YOLOv11 models retrained on datasets available on [Roboflow](https://roboflow.com/). Currently, there are three models used in the project:

1) Players and Referee (small dataset of 1,600 images)
2) Ball (trained on 15,000 images, but due to fast-moving balls, interpolation is often required)
3) Actions (still searching for a larger dataset, as the current model underperforms)

### Ideas:
- Court Detection: Likely using segmentation, with the goal of mapping player and ball coordinates to extract different statistics such as ball speed, player jump heights, movement heatmaps, setting patterns...
- Referee Pose Estimation: To detect referee decisions and identify the start and end of volleyball rallies, allowing trackers to pause between rallies...
