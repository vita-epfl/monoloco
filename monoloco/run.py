# pylint: disable=too-many-branches, too-many-statements, import-outside-toplevel

import argparse

from openpifpaf import decoder, network, visualizer, show, logger


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Subparser definition
    subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')
    predict_parser = subparsers.add_parser("predict")
    prep_parser = subparsers.add_parser("prep")
    training_parser = subparsers.add_parser("train")
    eval_parser = subparsers.add_parser("eval")

    # Predict (2D pose and/or 3D location from images)
    predict_parser.add_argument('images', nargs='*', help='input images')
    predict_parser.add_argument('--glob', help='glob expression for input images (for many images)')
    predict_parser.add_argument('--checkpoint', help='pifpaf model')
    predict_parser.add_argument('-o', '--output-directory', help='Output directory')
    predict_parser.add_argument('--output_types', nargs='+', default=['json'],
                                help='what to output: json keypoints skeleton for Pifpaf'
                                     'json bird front or multi for MonStereo')
    predict_parser.add_argument('--no_save', help='to show images', action='store_true')
    predict_parser.add_argument('--dpi', help='image resolution',  type=int, default=150)
    predict_parser.add_argument('--long-edge', default=None, type=int,
                                help='rescale the long side of the image (aspect ratio maintained)')
    predict_parser.add_argument('--white-overlay',
                                nargs='?', default=False, const=0.8, type=float,
                                help='increase contrast to annotations by making image whiter')
    predict_parser.add_argument('--font-size', default=0, type=int, help='annotation font size')
    predict_parser.add_argument('--monocolor-connections', default=False, action='store_true',
                                help='use a single color per instance')
    predict_parser.add_argument('--instance-threshold', type=float, default=None, help='threshold for entire instance')
    predict_parser.add_argument('--seed-threshold', type=float, default=0.5, help='threshold for single seed')
    predict_parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')
    predict_parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false',
                                help='use more exact image rescaling (requires scipy)')
    predict_parser.add_argument('--decoder-workers', default=None, type=int,
                                help='number of workers for pose decoding, 0 for windows')

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    predict_parser.add_argument('--mode', help='keypoints, mono, stereo', default='mono')
    predict_parser.add_argument('--model', help='path of MonoLoco/MonStereo model to load')
    predict_parser.add_argument('--net', help='only to select older MonoLoco model, otherwise use --mode')
    predict_parser.add_argument('--path_gt', help='path of json file with gt 3d localization')
    predict_parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=100)
    predict_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    predict_parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    predict_parser.add_argument('--show_all', help='only predict ground-truth matches or all', action='store_true')
    predict_parser.add_argument('--focal', help='focal length in mm for a sensor size of 7.2x5.4 mm. (nuScenes)',
                                type=float, default=5.7)

    # Social distancing and social interactions
    predict_parser.add_argument('--social_distance', help='social', action='store_true')
    predict_parser.add_argument('--threshold_prob', type=float, help='concordance for samples', default=0.25)
    predict_parser.add_argument('--threshold_dist', type=float, help='min distance of people', default=2.5)
    predict_parser.add_argument('--radii', type=tuple, help='o-space radii', default=(0.3, 0.5, 1))

    # Preprocess input data
    prep_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints', required=True)
    prep_parser.add_argument('--mode', help='mono, stereo', default='mono')
    prep_parser.add_argument('--dataset',
                             help='datasets to preprocess: nuscenes, nuscenes_teaser, nuscenes_mini, kitti',
                             default='kitti')
    prep_parser.add_argument('--dir_nuscenes', help='directory of nuscenes devkit', default='data/nuscenes/')
    prep_parser.add_argument('--iou_min', help='minimum iou to match ground truth', type=float, default=0.3)
    prep_parser.add_argument('--variance', help='new', action='store_true')
    prep_parser.add_argument('--activity', help='new', action='store_true')

    # Training
    training_parser.add_argument('--joints', help='Json file with input joints', required=True)
    training_parser.add_argument('--mode', help='mono, stereo', default='mono')
    training_parser.add_argument('--out', help='output_path, e.g., data/outputs/test.pkl')
    training_parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train for', default=500)
    training_parser.add_argument('--bs', type=int, default=512, help='input batch size')
    training_parser.add_argument('--monocular', help='whether to train monoloco', action='store_true')
    training_parser.add_argument('--dropout', type=float, help='dropout. Default no dropout', default=0.2)
    training_parser.add_argument('--lr', type=float, help='learning rate', default=0.002)
    training_parser.add_argument('--sched_step', type=float, help='scheduler step time (epochs)', default=30)
    training_parser.add_argument('--sched_gamma', type=float, help='Scheduler multiplication every step', default=0.98)
    training_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=1024)
    training_parser.add_argument('--n_stage', type=int, help='Number of stages in the model', default=3)
    training_parser.add_argument('--hyp', help='run hyperparameters tuning', action='store_true')
    training_parser.add_argument('--multiplier', type=int, help='Size of the grid of hyp search', default=1)
    training_parser.add_argument('--r_seed', type=int, help='specify the seed for training and hyp tuning', default=1)
    training_parser.add_argument('--print_loss', help='print training and validation losses', action='store_true')
    training_parser.add_argument('--auto_tune_mtl', help='whether to use uncertainty to autotune losses',
                                 action='store_true')
    training_parser.add_argument('--no_save', help='to not save model and log file', action='store_true')

    # Evaluation
    eval_parser.add_argument('--mode', help='mono, stereo', default='mono')
    eval_parser.add_argument('--dataset', help='datasets to evaluate, kitti or nuscenes', default='kitti')
    eval_parser.add_argument('--activity', help='evaluate activities', action='store_true')
    eval_parser.add_argument('--geometric', help='to evaluate geometric distance', action='store_true')
    eval_parser.add_argument('--generate', help='create txt files for KITTI evaluation', action='store_true')
    eval_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints (for KITTI evaluation)')
    eval_parser.add_argument('--model', help='path of MonoLoco model to load')
    eval_parser.add_argument('--joints', help='Json file with input joints to evaluate (for nuScenes evaluation)')
    eval_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    eval_parser.add_argument('--dropout', type=float, help='dropout. Default no dropout', default=0.2)
    eval_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=1024)
    eval_parser.add_argument('--n_stage', type=int, help='Number of stages in the model', default=3)
    eval_parser.add_argument('--show', help='whether to show statistic graphs', action='store_true')
    eval_parser.add_argument('--save', help='whether to save statistic graphs', action='store_true')
    eval_parser.add_argument('--verbose', help='verbosity of statistics', action='store_true')
    eval_parser.add_argument('--new', help='new', action='store_true')
    eval_parser.add_argument('--variance', help='evaluate keypoints variance', action='store_true')

    eval_parser.add_argument('--net', help='Choose network: monoloco, monoloco_p, monoloco_pp, monstereo')
    eval_parser.add_argument('--baselines', help='whether to evaluate stereo baselines', action='store_true')
    eval_parser.add_argument('--generate_official', help='whether to add empty txt files for official evaluation',
                             action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = cli()
    if args.command == 'predict':
        from .predict import predict
        predict(args)

    elif args.command == 'prep':
        if 'nuscenes' in args.dataset:
            from .prep.preprocess_nu import PreprocessNuscenes
            prep = PreprocessNuscenes(args.dir_ann, args.dir_nuscenes, args.dataset, args.iou_min)
            prep.run()
        else:
            from .prep.preprocess_kitti import PreprocessKitti
            prep = PreprocessKitti(args.dir_ann, mode=args.mode, iou_min=args.iou_min)
            if args.activity:
                prep.process_activity()
            else:
                prep.run()

    elif args.command == 'train':
        from .train import HypTuning
        if args.hyp:
            hyp_tuning = HypTuning(joints=args.joints, epochs=args.epochs,
                                   monocular=args.monocular, dropout=args.dropout,
                                   multiplier=args.multiplier, r_seed=args.r_seed)
            hyp_tuning.train(args)
        else:
            from .train import Trainer
            training = Trainer(args)
            _ = training.train()
            _ = training.evaluate()

    elif args.command == 'eval':
        if args.activity:
            from .eval.eval_activity import ActivityEvaluator
            evaluator = ActivityEvaluator(args)
            if 'collective' in args.dataset:
                evaluator.eval_collective()
            else:
                evaluator.eval_kitti()

        elif args.geometric:
            assert args.joints, "joints argument not provided"
            from .eval.geom_baseline import geometric_baseline
            geometric_baseline(args.joints)

        elif args.variance:
            from .eval.eval_variance import joints_variance
            joints_variance(args.joints, clusters=None, dic_ms=None)

        else:
            if args.generate:
                from .eval.generate_kitti import GenerateKitti
                kitti_txt = GenerateKitti(args)
                kitti_txt.run()

            if args.dataset == 'kitti':
                from .eval import EvalKitti
                kitti_eval = EvalKitti(args)
                kitti_eval.run()
                kitti_eval.printer()

            elif 'nuscenes' in args.dataset:
                from .train import Trainer
                training = Trainer(args)
                _ = training.evaluate(load=True, model=args.model, debug=False)

            else:
                raise ValueError("Option not recognized")

    else:
        raise ValueError("Main subparser not recognized or not provided")


if __name__ == '__main__':
    main()
