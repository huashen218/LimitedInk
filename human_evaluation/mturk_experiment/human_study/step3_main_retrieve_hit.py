import os
import json
import argparse
import xmltodict
import numpy as np
from mturk_cores import MTurkManager, print_log



def main(args):

    # ===============================
    # step1: Parsing Configurations
    # ===============================
    human_eval_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    html_file_folder = os.path.join(human_eval_root_dir, "user_interface", "human_study")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    worker_result_dir = os.path.join(human_eval_root_dir, "mturk_experiment", "recruit_participants", "MTurk_WorkerGroup_Track.json")

    """
    To test HIT in sandbox, set args.test_hit = True (default);
    If post HIT for production, please set args.test_hit = False.
    """
    config_dir = os.path.join(human_eval_root_dir, "mturk_experiment", "config_sandbox.json") if args.test_hit \
                else os.path.join(human_eval_root_dir, "mturk_experiment", "config.json")
    with open(config_dir, "r") as read_file:
        config = json.load(read_file)
    print_log("INFO", f" ====== Display Your Configuration ====== \n{config}")



    # ===============================
    # step2: MTurkManager Client Setup
    # ===============================
    """
    client functions to view:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk.html#MTurk.Client.create_worker_block
    """
    mturk_manager = MTurkManager(config)
    mturk_manager.set_environment()
    mturk_manager.setup_client()
    client = mturk_manager.client
    print(client.get_account_balance()['AvailableBalance'])




    # ===============================
    # step3: Retrieve and Approve MTurk Results
    # ===============================
    result_id = "11_11_2021_11_05"   # This result_id should be identical to the date_time in `step1_main_create_hit.py`
    save_results_dir = os.path.join(current_dir, f"human_study_sandbox_{result_id}.json") if args.test_hit \
                    else  os.path.join(current_dir, f"human_study_production_{result_id}.json")
    with open(save_results_dir, "r") as read_file:
        results = json.load(read_file)
    result_json_file_dir = os.path.join(current_dir, f"MTurk_results_{result_id}.json")


    all_results = {}
    batch_number, group_number = 20, 10
    count = 0
    for g in range(group_number):
        batches = results[f'WorkerGroup{g}']
        for item in batches:
            count += 1
            hit = client.get_hit(HITId=item['hit_id'])
            all_results[item['hit_id']] = {}
            all_results[item['hit_id']]["batch"] = item['batch']
            all_results[item['hit_id']]["dataset_idx"] = item['dataset_idx']


            assignmentsList = client.list_assignments_for_hit(
                HITId=item['hit_id'],
                AssignmentStatuses=['Submitted', 'Approved'],
                MaxResults=100
            )

            assignments = assignmentsList['Assignments']
            item['assignments_submitted_count'] = len(assignments)
            print(f" ### No.{count}: HIT={item['hit_id']}; Submitted Assignments Number = {item['assignments_submitted_count']} ### ")
            

            answers = []
            for assignment in assignments:

                answer_dict = xmltodict.parse(assignment['Answer'])
                answer = answer_dict['QuestionFormAnswers']['Answer']
                each_answer = {}
                each_answer["WorkerID"] = assignment['WorkerId']
                for ans in answer:
                    each_answer[ans['QuestionIdentifier']] = ans['FreeText']
                answers.append(each_answer)

                if assignment['AssignmentStatus'] == 'Submitted':
                    client.approve_assignment(
                        AssignmentId=assignment['AssignmentId'],
                        OverrideRejection=False
                    )

            all_results[item['hit_id']]["Answers"] = answers

    with open(result_json_file_dir, 'w') as json_file:
        json.dump(all_results, json_file, indent = 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_hit', action='store_true', dest='test_hit', default=True)
    args = parser.parse_args()
    main(args)




