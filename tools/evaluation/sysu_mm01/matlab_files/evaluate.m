function evaluate(workspace, feature_dir, result_dir, mode, number_shot)

    feature_info.name = 'feature';
    feature_info.dir = feature_dir
    result_dir = result_dir
    % feature_info.name = 'feature';
    % feature_info.dir = './feature';
    % result_dir = './result'; % directory for saving result

    setting.mode = mode; %'all_search' 'indoor_search'
    setting.number_shot = number_shot; % 1 for single shot, 10 for multi-shot
    % setting.mode = 'all_search'; %'all_search' 'indoor_search'
    % setting.number_shot = 1; % 1 for single shot, 10 for multi-shot

    model.test_fun = @euclidean_dist; % Similarity measurement function
    model.name = 'euclidean'; % model name
    model.para = []; % No parameter is needed for euclidean distance here

    % load data split
    content = load([workspace, 'data_split/test_id.mat']); % fixed testing person IDs
    data_split.test_id = content.id;

    content = load([workspace, 'data_split/rand_perm_cam.mat']); % fixed permutation of samples in each camera
    data_split.rand_perm_cam = content.rand_perm_cam;

    % evaluation
    performance = evaluation_SYSU_MM01(feature_info, data_split, model, setting, result_dir);
