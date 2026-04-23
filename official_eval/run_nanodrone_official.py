import nonlinear_benchmarks
from nonlinear_benchmarks.nanodrone_error_metrics import compute_errors, print_results

MAX_H = 50

def predict_one_run(model, test_run):
    """
    你自己实现这个函数：
    输入：你训练好的模型、一个官方 test_run
    输出：一个 DataFrame，至少包含：
      t,
      x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,
      以及 {state}_pred_h{1..50}
    """
    raise NotImplementedError

def main():
    train_sets, test_sets = nonlinear_benchmarks.NanoDrone()

    # 这里不要用 test_sets 调模型/选超参
    # 你可以在这里加载你已经训练好的最终模型
    model = None  # TODO: load your trained model

    all_metrics = []
    for test_run in test_sets:
        df_pred = predict_one_run(model, test_run)
        metrics = compute_errors(df_pred, max_horizon=MAX_H)
        print_results(metrics, label=test_run.name, max_horizon=MAX_H)
        all_metrics.append(metrics)

if __name__ == "__main__":
    main()