import sys


def test_distill_accepts_teachers_flag():
    # Smoke: verify CLI parses --teachers and runs main up to data/teacher load
    from omnicoder.training.distill import main as kd_main
    argv = sys.argv
    try:
        sys.argv = [
            'distill', '--data', '.', '--seq_len', '8', '--steps', '1', '--device', 'cpu',
            '--student_mobile_preset', 'mobile_4gb', '--teachers', 'sshleifer/tiny-gpt2', 'sshleifer/tiny-gpt2'
        ]
        # Expect it to execute without raising; actual teacher load may fallback to CPU
        kd_main()
    finally:
        sys.argv = argv


