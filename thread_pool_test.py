import concurrent.futures

import time

def task(n):
    print(f"Task {n} is starting")
    time.sleep(2)
    print(f"Task {n} is completed")
    return n * 2

def main():
    tasks = [1, 2, 3, 4, 5]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {executor.submit(task, n): n for n in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task_number = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Task {task_number} returned {result}")
            except Exception as exc:
                print(f"Task {task_number} generated an exception: {exc}")
    print("All tasks completed.")


if __name__ == "__main__":
    main()
    