from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, ActionTracker
from team_assigner import TeamAssigner


def main():
    # Input Video
    input_video_path = 'input_videos/test4.mp4'
    video_frames = read_video(input_video_path)

    # Initialize trackers
    people_tracker = PlayerTracker(model_path='models/player_referee_best.pt') # was player_det.pt
    ball_tracker = BallTracker(model_path='models/ball_retrained.pt')
    action_tracker = ActionTracker(model_path='models/action_last.pt')

    # Detect people and separate them as players and referee
    people_detections = people_tracker.detect_frames(video_frames, read_from_stub=False, stub_path='tracker_stubs/player_referee_detections.pkl')
    player_detections = people_detections['player']
    referee_detections = people_detections['referee']
    
    # Detect ball
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path='tracker_stubs/ball_detections.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect volleyball action
    action_detections = action_tracker.detect_frames(video_frames, read_from_stub=False, stub_path='tracker_stubs/action_detections.pkl')

    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.find_middle_line(video_frames, 0)

    for frame_num, player_track in enumerate(player_detections):
        for player_id, track in player_track.items():
            color = team_assigner.assign_color(track['bbox'])
            player_detections[frame_num][player_id]['team_color'] = color

    
    # Anotate video
    output_video_frames = people_tracker.draw_ellipses(video_frames, player_detections)
    output_video_frames = people_tracker.anotate_referee(video_frames, referee_detections)
    output_video_frames = action_tracker.draw_actions_info(video_frames, action_detections)
    output_video_frames = ball_tracker.draw_ball_path(video_frames, ball_detections)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video76.mp4')

if __name__ == '__main__':
    main()