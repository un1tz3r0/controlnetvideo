"""
Shared pytest fixtures and configuration for the test suite.
"""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_image():
    """Mock PIL Image object."""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a small test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    except ImportError:
        # Return a mock object if PIL is not available
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        return mock_img


@pytest.fixture
def mock_video_frames():
    """Mock video frames as a list of PIL Images."""
    try:
        from PIL import Image
        import numpy as np
        
        frames = []
        for i in range(5):  # 5 frame mock video
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frames.append(Image.fromarray(img_array))
        return frames
    except ImportError:
        # Return mock objects if PIL/numpy is not available
        frames = []
        for i in range(5):
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            frames.append(mock_img)
        return frames


@pytest.fixture
def mock_torch_device():
    """Mock torch device selection."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.device', return_value='cpu'):
            yield 'cpu'


@pytest.fixture
def mock_model_loading():
    """Mock model loading to avoid downloading large models."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    
    # Just return the mock model without trying to patch non-existent modules
    # In real usage, this would patch the actual imports when the dependencies are installed
    yield mock_model


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        'model_id': 'test/model',
        'controlnet_model_id': 'test/controlnet', 
        'device': 'cpu',
        'num_inference_steps': 20,
        'guidance_scale': 7.5,
        'controlnet_conditioning_scale': 1.0
    }


@pytest.fixture
def mock_file_system(temp_dir):
    """Mock file system operations."""
    test_files = {
        'input.jpg': temp_dir / 'input.jpg',
        'output.jpg': temp_dir / 'output.jpg',
        'config.json': temp_dir / 'config.json'
    }
    
    # Create some test files
    for filename, filepath in test_files.items():
        filepath.touch()
    
    return test_files


@pytest.fixture
def mock_opencv():
    """Mock OpenCV operations."""
    import numpy as np
    
    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mock_cv2.imwrite.return_value = True
    mock_cv2.VideoCapture.return_value.__enter__.return_value.read.return_value = (True, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    with patch('cv2', mock_cv2):
        yield mock_cv2


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def mock_cli_context():
    """Mock Click CLI context for command testing."""
    from unittest.mock import MagicMock
    
    ctx = MagicMock()
    ctx.params = {}
    ctx.get_parameter_source.return_value = 'commandline'
    return ctx