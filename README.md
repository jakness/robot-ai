# robot-ai

## Setup the environment

Create a conda virtual environment and activate it, e.g. with [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main):
```bash
conda create -y -n robot-ai python=3.10
conda activate robot-ai
```

Install `ffmpeg`:
```bash
  conda install ffmpeg -c conda-forge
```

Install `PyTorch`:
```bash
  conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install `robot-ai`:
```bash
  pip install -e ".[dev]"
```

For development, install the pre-commit hooks:
```bash
   pre-commit install
```

## Make tea

I use LLMs as the brain of the robot to decide what skills to use and in what order to complete the task of making tea.
LLMs make it possible to generalize the tea making task in a way that the same principles can be applied to other tasks
as long as the skills needed to complete a task have been trained.

There are two ways to make tea: using an agentic and a non-agentic approach, which still uses an LLM but is a bit more
deterministic than the agentic way. The non-agentic way starts with an LLM deciphering a task plan from user-defined
instructions to make tea. The task plan contains the needed robot skills in order to make tea. The task plan can be
checked for correctness, and if it is correct, we can just loop through the skills in the task plan and be sure that
the skills used are correct and executed in the right order. On the other hand, the agentic way uses an LLM to iteratively
decide what skills to use, which introduces a lot more uncertainty in the execution order. In this case, the agentic
approach doesn't really provide any benefits over the non-agentic way and was more of an experiment on how well the 
agent (LLM) follows instructions.

One important detail for generalization is the fact that we need to know if the skill usage was a success or not before
continuing to execute the next skill. For this, we use the video analysis capabilities of an LLM, which generalizes
to every skill. You just need to ask the right question to determine whether the skill was executed successfully.
Without the video analysis capabilities of an LLM, we would need to build a custom solution for each skill to determine
the success of the skill execution, and depending on the skill, this could be quite difficult to do.

I tried the `gemini-2.5-pro` and `gemini-2.5-flash` models from Google as the LLM. Deciphering the task plan in the non-agentic way
was easy for both models and never failed. Video analysis was kind of hard for them. `gemini-2.5-pro`
was better at it but still failed sometimes to analyze the video correctly. The agent with `gemini-2.5-pro` as the model
worked surprisingly well and was able to complete the task of making tea quite often, but sometimes it failed to
execute the skills in the right order. The agent with `gemini-2.5-flash` as the model, on the other hand, failed to
complete the task of making tea every time. Sometimes it seemed that it's going to complete the task, but then it
once again blundered and failed to complete the task.

### Environment variables
We are using LLMs from Google. Add Gemini's API key to the environment variable `GEMINI_API_KEY`

Tea is made using four skills. A model needs to be trained for each skill using the LeRobot library.
Add the trained model paths for each skill to the environment variables:
* `PICK_TEABAG_DROP_IN_CUP`
* `REMOVE_TEABAG_FROM_CUP`
* `SUGAR_CUBE_IN_CUP`
* `STIR_SPOON`

### Non-agentic way
```bash
  python -m robot_ai.tea_maker.make_tea
```

### Agent with function tools
```bash
  python -m robot_ai.tea_maker.tea_making_agent
```

### Agent and MCP server
```bash
  # Start the MCP server
  python -m robot_ai.tea_maker.mcp_server

  # Start the agent
  python -m robot_ai.tea_maker.mcp_tea_making_agent
```

### Notes

#### Data collection
Try to set up your environment in a way that minimizes variation so you don't need to collect too much data.
For example, try to use your camera in the same position and orientation for each data collection session. In my case,
this wasn't possible, but I knew the limits of the position and the orientation of the camera, so I made sure to change
the camera position and orientation within those limits in each data collection session to catch the variation in my
camera setup. This same principle applies to everything else in your environment, such as the position of the teacup
or the position of the sugar cubes, etc.

* Pick up a teabag and drop it in a teacup
  * This is a pretty simple pick-and-place task. Of course, if the teabags are scattered around, you need to make sure
  you take this variation into account. I didn't have that much variation, so I was able to get a pretty good model
  using 50 episodes of data collection.
* Remove the teabag from the teacup
  * Even though this is also a simple pick-and-place task, it was the hardest task to learn. At first, I wanted to
  test how well could the model be trained to lift the teabag by the string, but this turned out to be
  too difficult since the string of the teabag is very thin and requires quite a lot of precision to grab it. The thin
  string is hard to see in the camera even for a human, so the model has a hard time learning to use the image features
  for such a precise task. I collected 50 episodes, and maybe the model would have worked better with more data, but
  it didn't seem likely, so I changed the way I grasped the teabag to a more manageable way. I also noticed that
  there is quite a lot of variation in the position of the teabag in the teacup after it has been dropped into it, so I
  made sure to collect the new dataset with a different grasping technique in a way that the position of the teabag also
  varies. The datasets had a lot of overlap, like how to approach the teabag and where to place the removed teabag, so
  the model had already learned a lot from the first dataset, which is why I continued to train the model on this new
  dataset. The position of the teabag after dropping it into the teacup was quite random, so I collected a bit more data -
  60 episodes - to capture the variation in the position of the teabag. Sometimes the teabag dropped into a position that
  wasn't reachable by the robot without moving the teacup, but moving the teacup would have introduced even more
  variation in the environment, so I decided to ignore these cases and not teach the robot to move the teacup.
* Drop a sugar cube in the teacup
  *  I had a bowl of sugar cubes, and naturally, the sugar cubes were randomly scattered in the bowl. When collecting
  data, I made sure to collect sugar cubes from different positions in the bowl to teach the model to better use the
  image features associated with the sugar cubes. Sometimes the model wasn't successful in picking up the sugar, but
  it still tried to drop the imagined sugar cube in the teacup. To mitigate this issue, I collected data where I
  unsuccessfully picked a sugar cube and started to move it toward the teacup, but in the middle of moving it, I
  "noticed" that I hadn't picked up a sugar cube and went back to the bowl to try again.
* Stir with the spoon inside the teacup
  * There is a design choice between the prediction horizon of the model and how you collect the data for this task.
  I like to stir in a circular motion, which means that the cycles are going to be pretty similar to each other, and
  the similarity of the cycles is going to affect the choice of the prediction horizon for the model or how long you 
  want the stirring to last. If you stir for a longer time than the prediction horizon, the model might not be able
  to learn to stop stirring and just keeps repeating the stirring cycle indefinitely. So, you need to be aware of
  the choice of the prediction horizon and how long you want the stirring to last. Otherwise, I just needed to make
  sure to collect data with different positions of the spoon in the teacup, since the position of the spoon in the
  teacup varies naturally.

#### Models
* ACT (https://arxiv.org/pdf/2304.13705)
  * As suggested in the ACT paper, action chunking and temporal ensembling made the
  model work better. Without temporal ensembling, the model sometimes started to predict actions very close to each other,
  which led the robot to get stuck in one place. Using the default value for temporal ensembling from the 
  original ACT work makes the model work well and the robot to move smoothly and not get stuck in one place. For the
  action chunks, I decided to go with 2 seconds worth of chunks. I didn't really need to experiment more with different
  chunk sizes since the 2 seconds worth of chunks worked well. Choosing a prediction horizon of 2 seconds affected the
  data collection for the stirring task. At first, I stirred for approximately 4 seconds, and because the prediction
  horizon was shorter, the model didn't really learn to stop stirring and just kept repeating the stirring cycle 
  indefinitely. I collected more data where I stirred for approximately 1.5 seconds, which helped the model to learn to
  stop stirring and finish the task.
  * Important parameters used
    * n_obs_steps: 1
      * History length
    * chunk_size: 60
      * With 30 FPS video, this corresponds to 2 seconds worth of actions
    * n_action_steps: 1
      * At every inference step run one action. Needs to be 1 to use the temporal ensembling.
    * temporal_ensemble_coeff: 0.01
* Diffusion policy (https://arxiv.org/pdf/2303.04137)
  * I didn't have a powerful enough computer to test the diffusion policy properly. At every step, the inference
  took a while, so the robot moved a couple of steps ahead, stopped for a while, moved a couple of steps ahead, and so on.
  This was too slow and annoying for testing, so I only tried to train the model for the stirring task since
  the stirring is inherently multi-modal, and the diffusion policy should work well for multi-modal tasks. As we remember,
  ACT didn't handle the stirring task well except when the prediction horizon and the length of the stirring time were
  taken into account. I used the same 2-second horizon as in ACT for the model, except for the diffusion
  policy I needed to split the 2 seconds into 1 second of observations from history and 1 second's worth of future
  predictions. I wanted a bit more observations from history, but my GPU didn't have enough memory, so I needed to
  extend the prediction horizon to have a total horizon of 2 seconds. I used the data where I stirred for 4 seconds
  to train the model and tested the model to see that it was able to stop the stirring motion, which it was, even though
  the data was collected with a longer window than the working horizon of the model. At the beginning of every inference
  step, the diffusion policy samples an action from a standard Gaussian distribution and starts to denoise the action to
  get the final prediction. This sampling helps the policy break from the stirring motion because the sampled action
  can come from the actions where we start to end the stirring motion or from the actions where we continue stirring.
  * Important parameters used
    * n_obs_steps: 30
      * History length
    * horizon: 64
      * Prediction horizon, but the LeRobot code is built in a way that in practice maximum of horizon - n_obs_steps = 64 - 30 = 34
      predicted actions can be used.
    * n_action_steps: 8
      * At every inference step run 8 actions.

### Example execution
* https://youtube.com/shorts/xu5SK6N3HO8
  * Dropping the teabag into the teacup and removing it were performed successfully. When trying to drop the sugar cube in
  the teacup, we can notice that at first the robot failed to grab a sugar cube, but then it went back to grab a new
  sugar cube and was able to grab one and drop it in the teacup successfully. This is an example of what was
  talked about in the data collection section. Stirring failed at first because the robot failed to grab the spoon but
  there is the component for checking whether the skill usage was successful or not, and in this case it noticed that
  the skill usage wasn't successful, which resulted in the robot trying to perform the skill again. Stirring the spoon
  the second time was done successfully.