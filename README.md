# sml_face_detection

# Requirement

- Install:

```markdown
pip install -r requirement
```

# Freeze
- Download weight at [here](https://drive.google.com/drive/folders/1sqHS_tWQM7AzdgRrvcHE0htZw9nngoYr?usp=sharing)
- Put weight at folder: `data/`

# How to use:
```sh
    $ git clone https://github.com/trung1309vn/sml_face_detection.git

    $ cd sml_face_detection

    $ python run_face.py
```

* Note 1: arguments above are specific for video's contain folder and name, change it in main function for your case, also output folder

* Note 2: According to the code, results are saved in database folder, which is included "duration_container_i.npy" - period of time people is detected in region i, "emotion_container_i.npy" - emotion sequence of people in region i, "face_container_i.npy" - pics of people detected in region i, "num_of_face_i" - number of people in region i

* Note 3: People detected in each region using yolov3 face detection and stored in database as one person if and only the distance between two occupied frames in that region is atmost time_to_live frames. If not the system will count it as new person 