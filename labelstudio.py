from label_studio_sdk import Client
import pandas as pd

def get_bounding_boxes(project_id:int = 1):
    """ access the label studio instance and download the boxes
        returns an array of dataframes. Each dataframe is a different whale.
    """
    # Define the URL where Label Studio is accessible and the API key for your user account
    LABEL_STUDIO_URL = 'https://label.elliotnas.synology.me'
    API_KEY = '4df35d9de2ebc568426662c114fba972d22fc821'
    PROJECT_ID = 1

    # Connect to the Label Studio API and check the connection
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()

    # get the project and labelling tasks (videos)
    project = ls.get_project(project_id)
    tasks = project.get_tasks()

    # this gets the boxes per whale, but theyre only the manually annotated timestamps, and not every frame
    whale_boxes = []
    for task in tasks:
        for annotation in task['annotations']:
            boxes = annotation['result']
            for box in boxes:
                sequence = box['value']['sequence']

                sequence_df = pd.DataFrame(sequence)

                whale_boxes.append(sequence_df)

    # interpolate the boxes over every frame
    interpolated_whale_boxes = []
    for sequence_df in whale_boxes:
        start_frame = sequence_df["frame"].min()
        end_frame = sequence_df["frame"].max()
        sequence_df.set_index('frame', inplace=True)

        sequence_df = sequence_df.reindex(pd.RangeIndex(start=start_frame, stop=end_frame + 1, step=1))

        sequence_df['enabled'] = sequence_df['enabled'].ffill()
        sequence_df[['rotation', 'x', 'y', 'width', 'height', 'time']] = sequence_df[
            ['rotation', 'x', 'y', 'width', 'height', 'time']].interpolate()

        interpolated_whale_boxes.append(sequence_df)

    return interpolated_whale_boxes
