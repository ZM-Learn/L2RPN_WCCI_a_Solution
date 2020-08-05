import os
import shutil
import io
import sys
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import argparse

from grid2op.Episode import EpisodeReplay

SCORE_TXT = "scores.txt"
RESULT_HTML = "results.html"
META_JSON = "episode_meta.json"
TIME_JSON = "episode_times.json"
REWARD_JSON = "other_rewards.json"

def create_fig(title, x=5, y=2, width=1280, height=720, dpi=96):
    w = width / dpi
    h = height / dpi
    fig, axs = plt.subplots(ncols=x, nrows=y, figsize=(w, h), sharey=True)
    fig.suptitle(title)
    return fig, axs

def draw_steps_fig(ax, step_data, ep_score, ep_share):
    nb_timestep_played = int(step_data["nb_timestep_played"])
    chronics_max_timestep = int(step_data["chronics_max_timestep"])
    n_blackout_steps = chronics_max_timestep - nb_timestep_played
    title_fmt = "Scenario {}\n({:.2f}/{:.2f})"
    scenario_dir = os.path.basename(step_data["chronics_path"])
    scenario_name = title_fmt.format(scenario_dir, ep_score, ep_share)
    labels = 'Played', 'Blackout'
    colors = ['blue', 'orange']
    fracs = [
        nb_timestep_played,
        n_blackout_steps
    ]
    def pct_fn(pct):
        n_steps = int(pct * 0.01 * chronics_max_timestep)
        return "{:.1f}%\n({:d})".format(pct, n_steps)
    ax.pie(fracs, labels=labels,
           autopct=pct_fn,
           startangle=90.0)
    ax.set_title(scenario_name)

def draw_rewards_fig(ax, reward_data, step_data):
    nb_timestep_played = int(step_data["nb_timestep_played"])
    chronics_max_timestep = int(step_data["chronics_max_timestep"])
    n_blackout_steps = chronics_max_timestep - nb_timestep_played
    scenario_name = "Scenario " + os.path.basename(step_data["chronics_path"])

    x = list(range(chronics_max_timestep))
    n_rewards = len(reward_data[0].keys())
    y = [[] * n_rewards]
    labels = list(reward_data[0].keys())
    for rel in reward_data:
        for i, v in enumerate(rel.values()):
            y[i].append(v)
    for i in range(n_rewards):
        y[i] += [0.0] * n_blackout_steps

    for i in range(n_rewards):
        ax.plot(x, y[i], label=labels[i])
    ax.set_title(scenario_name)
    ax.legend()

def fig_to_b64(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    fig_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return fig_b64

def html_result(score, duration, fig_list):
    html = """<html><head></head><body>\n"""
    html += """<div style='margin: 0 auto; width: 500px;'>"""
    html += """<p>Score {}</p>""".format(np.round(score, 3))
    html += """<p>Duration {}</p>""".format(np.round(duration, 2))
    html += """</div>"""
    for i, figure in enumerate(fig_list):
        html += '<img src="data:image/png;base64,{0}"><br>'.format(figure)
    html += """</body></html>"""
    return html

def html_error():
    html = """<html><head></head><body>\n"""
    html += """Invalid submission"""
    html += """</body></html>"""
    return html

def cli():
    DEFAULT_TIMEOUT_SECONDS = 20*60
    DEFAULT_NB_EPISODE = 10
    DEFAULT_KEY_SCORE = "tmp_score_codalab"
    DEFAULT_GIF_EPISODE = None
    DEFAULT_GIF_START = 0
    DEFAULT_GIF_END = 50

    parser = argparse.ArgumentParser(description="Scoring program")
    parser.add_argument("--logs_in", required=True,
                        help="Path to the runned output directory")
    parser.add_argument("--config_in", required=True,
                        help="DoNothing json config input file")
    parser.add_argument("--data_out", required=True,
                        help="Path to the results output directory")
    parser.add_argument("--key_score", required=False,
                        default=DEFAULT_KEY_SCORE, type=str,
                        help="Codalab other_reward name")
    parser.add_argument("--timeout_seconds", required=False,
                        default=DEFAULT_TIMEOUT_SECONDS, type=int,
                        help="Number of seconds before codalab timeouts")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes in logs in")
    parser.add_argument("--gif_episode", required=False,
                        default=DEFAULT_GIF_EPISODE, type=str,
                        help="Name of the episode to generate a gif for")
    parser.add_argument("--gif_start", required=False,
                        default=DEFAULT_GIF_START, type=int,
                        help="Start step for gif generation")
    parser.add_argument("--gif_end", required=False,
                        default=DEFAULT_GIF_END, type=int,
                        help="End step for gif generation")
    return parser.parse_args()

def write_output(output_dir, html_content, score, duration):
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Write scores
    score_filename = os.path.join(output_dir, SCORE_TXT)
    with open(score_filename, 'w') as f:
        f.write("score: {:.6f}\n".format(score))
        f.write("duration: {:.6f}\n".format(duration))

    # Write results
    result_filename = os.path.join(output_dir, RESULT_HTML)
    with open(result_filename, 'w') as f:
        f.write(html_content)

def write_gif(output_dir, agent_path, episode_name, start_step, end_step):
    epr = EpisodeReplay(agent_path)
    epr.replay_episode(episode_name, fps=2.0, display=False,
                       gif_name=episode_name,
                       start_step=start_step, end_step=end_step)
    gif_genpath = os.path.join(agent_path, episode_name, episode_name + ".gif")
    gif_outpath = os.path.join(output_dir, episode_name + ".gif")
    print (gif_genpath, gif_outpath)
    if os.path.exists(gif_genpath):
        shutil.move(gif_genpath, gif_outpath)

def compute_episode_score(key_score,
                          meta, other_rewards,
                          score_conf, ep_info):
    n_steps = int(meta["chronics_max_timestep"])
    n_played = int(meta["nb_timestep_played"])
    ep_loads = np.array(ep_info["sum_loads"])
    ep_losses = np.array(ep_info["losses"])
    ep_marginal_cost = ep_info["marginal_cost"]
    min_losses_ratio = score_conf["min_losses_ratio"]
    ep_do_nothing_reward = ep_info["donothing_reward"]
    ep_do_nothing_nodisc_reward = ep_info["donothing_nodisc_reward"]

    # Compute cumulated reward
    ep_reward = 0.0
    for rel in other_rewards:
        ep_reward += rel[key_score]

    # Add cost of non delivered loads for blackout steps
    blackout_loads = ep_loads[n_played:]
    if len(blackout_loads) > 0:
        blackout_reward = np.sum(blackout_loads) * ep_marginal_cost
        ep_reward += blackout_reward

    # Compute ranges
    worst_reward = np.sum(ep_loads) * ep_marginal_cost
    best_reward = np.sum(ep_losses) * min_losses_ratio
    zero_reward = ep_do_nothing_reward
    zero_blackout = ep_loads[ep_info["dn_played"]:]
    zero_reward += np.sum(zero_blackout) * ep_marginal_cost
    nodisc_reward = ep_do_nothing_nodisc_reward

    # Linear interp episode reward to codalab score
    if zero_reward != nodisc_reward:
        # DoNothing agent doesnt complete the scenario
        reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
        score_range = [100.0, 80.0, 0.0, -100.0]
    else:
        # DoNothing agent can complete the scenario
        reward_range = [best_reward, zero_reward, worst_reward]
        score_range = [100.0, 0.0, -100.0]
        
    ep_score = np.interp(ep_reward, reward_range, score_range)
    return ep_score

def main():
    args = cli()
    input_dir = args.logs_in
    output_dir = args.data_out
    config_file = args.config_in

    print("input dir: {}".format(input_dir))
    print("output dir: {}".format(output_dir))
    print("config json: {}".format(config_file))
    print("input content", os.listdir(input_dir))

    with open(config_file, "r") as f:
        config = json.load(f)

    # Fail if input doesn't exists
    if not os.path.exists(input_dir):
        error_score = -100.0
        error_duration = args.timeout_seconds + 1
        write_output(output_dir, html_error(), error_score, error_duration)
        sys.exit("Your submission is not valid.")

    # Create output variables
    total_duration = 0.0
    total_score = 0.0
    ## Create output figures 
    step_w = 5
    step_h = max(args.nb_episode // step_w, 1)
    step_fig, step_axs = create_fig("Completion", x=step_w, y=step_h)
    reward_w = 2
    reward_h = max(args.nb_episode // reward_w, 1)
    reward_title = "Cost of grid operation & Custom rewards"
    reward_fig, reward_axs = create_fig(reward_title,
                                        x=reward_w, y=reward_h,
                                        height=1750)
    episode_index = 0
    episode_names = config["episodes_info"].keys()
    score_config = config["score_config"]
    for episode_id in sorted(episode_names):
        # Get info from config
        episode_info = config["episodes_info"][episode_id]
        episode_len = float(episode_info["length"])
        episode_weight =  episode_len / float(score_config["total_timesteps"])

        # Compute episode files paths
        scenario_dir = os.path.join(input_dir, episode_id)
        meta_json = os.path.join(scenario_dir, META_JSON)
        time_json = os.path.join(scenario_dir, TIME_JSON)
        reward_json = os.path.join(scenario_dir, REWARD_JSON)
        if not os.path.isdir(scenario_dir) or \
           not os.path.exists(meta_json) or \
           not os.path.exists(time_json) or \
           not os.path.exists(reward_json):
            episode_score = -100.0
            episode_index += 1
            total_score += episode_weight * episode_score
            continue

        with open(meta_json, "r") as f:
            meta = json.load(f)
        with open(reward_json, "r") as f:
            other_rewards = json.load(f)
        with open(time_json, "r") as f:
            timings = json.load(f)

        episode_score = compute_episode_score(args.key_score,
                                              meta, other_rewards,
                                              score_config, episode_info)
        # Draw figs
        step_ax_x = episode_index % step_w
        step_ax_y = episode_index // step_w
        draw_steps_fig(step_axs[step_ax_y, step_ax_x],
                       meta, episode_score * episode_weight,
                       episode_weight * 100.0)
        reward_ax_x = episode_index % reward_w
        reward_ax_y = episode_index // reward_w
        draw_rewards_fig(reward_axs[reward_ax_y, reward_ax_x],
                         other_rewards, meta)

        # Loop to next episode
        episode_index += 1
        # Sum durations and scores
        total_duration += float(timings["Agent"]["total"])
        total_score += episode_weight * episode_score

    # Format result html page
    step_figb64 = fig_to_b64(step_fig)
    reward_figb64 = fig_to_b64(reward_fig)
    html_out = html_result(total_score, total_duration,
                           [step_figb64, reward_figb64])

    # Write final output
    write_output(output_dir, html_out, total_score, total_duration)

    # Generate a gif if enabled
    if args.gif_episode is not None:
        write_gif(output_dir, input_dir, args.gif_episode,
                  args.gif_start, args.gif_end)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("ERROR: scoring program failed with error: \n{}".format(e))
        print("Traceback is:")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
