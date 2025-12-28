import traceback
from controller import EyeGestureController

if __name__ == "__main__":
    controller = None
    try:
        controller = EyeGestureController()
        controller.run()
    except Exception as e:
        print(f"Erreur Critique: {e}")
        traceback.print_exc()
    finally:
        if controller is not None:
            try:
                controller.shutdown()
            except Exception:
                pass
