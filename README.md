# SNU URP21 hexapod project

이 repository에서는, PhantomAX mk2 urdf 를 기반으로 한 pybullet gym env 및 그 train 파일을 다룬다.

## urdf 기반 gym env 구조에 대한 설명

https://github.com/GerardMaggiolino/Gym-Medium-Post: urdf 기반의 단순한 gym env 및 TRPO 학습 example
https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e: 관련 포스팅

위 repo와 포스팅은 gym env 구조의 파악에 도움이 되는 자료이다.
<br>agent.py는 SB3의 일부 기능에 해당하고, main.py는 train_first.py(우리가 쓰던)에 해당한다.

urdf 기반의 gym env의 디렉토리 구조의 개략도는 아래와 같다.

    project_dir/
        setup.py (이 env를 패키지화해주어서, train_first 등의 py에서 import 가능하게 해줌)
        env_name/
            __init__.py (gym env에 본 env를 register해주는 과정, mujoco때도 했다.)
            envs/
                __init__.py (패키지화된 본 env에서 아래 파일의 class를 import)
                env_name_env.py (진짜 env. 여기서 최소한 __ init __ 함수, reset 함수, step 함수, render 함수, seed 함수, close 함수는 정의되어야 한다. 추가적인 함수 추가도 가능. 지금까지 mujoco에서 다루던 hexy_v4.env에 해당한다. 다만, 보다 직접적으로 더 많은 것을 지정해줘야 함. 오히려 장점?)
            rsc/
                __init__ .py (필요없음)
                rsc1.py (urdf를 불러오고, 필요하다면 action 부여도 가능하게)
                rsc1.urdf
                rsc2.py
                rsc2.urdf
                ...

(후처리로, project_dir에서의 pip install -e . 가 필요할 수도 있다)
본 repository는 이와 유사한 구도를 가지고 있다. 

더이상 기존 라이브러리 파일을 수정하지 않고도 별도의 디렉토리 관리가 가능해졌고,
이를 통해 더 복잡한 env및 train을 설계하는 것이 가능할 것이다.

## train 관련 파일에 대하여

현재로서는 stable baselines 3 를 그대로 사용하고 있다.
