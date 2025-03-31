import unittest
import torch
from dataset import GCSDataLoader
from model import GCSModel
from callbacks import SaturationVisualizationCallback
# import matplotlib.pyplot as plt

class TestGCSDataLoader(unittest.TestCase):
    def setUp(self):
        self.k_file = "/home/ubuntu/DeFINO/data_generation/src/rescaled_200_fields.h5"
        self.states_dir = "/home/ubuntu/DeFINO/data_generation/src"
        self.vjp_dir = "/home/ubuntu/DeFINO/data_generation/src/num_ev_8"
        self.eigvec_dir = "/home/ubuntu/DeFINO/data_generation/src/num_ev_8"

        self.data_loader = GCSDataLoader(
            k_file=self.k_file,
            states_dir=self.states_dir,
            batch_size=1,
            nt=5,
            nx=256,
            ny=256,
            num_vec=8,    
            vjp_dir=self.vjp_dir,
            eigvec_dir=self.eigvec_dir,
        )

        self.train_loader = self.data_loader.get_dataloader(offset=0, limit=1)

    def test_batch_shapes(self):
        for batch in self.train_loader:
            self.assertEqual(batch['x'].shape, (1, 1, 5, 256, 256))
            self.assertEqual(batch['y'].shape, (1, 1, 5, 256, 256))
            self.assertEqual(batch['vjp'].shape, (1, 8, 5, 256, 256))
            self.assertEqual(batch['eigvec'].shape, (1, 8, 5, 256, 256))
            break

            # Visualization code for reference:
            # Plot saturation fields for each timestep
            # for i in range(batch['y'].shape[1]):
            #     temp = SaturationVisualizationCallback().plot_results(batch['y'][0][i], batch['y'][0][i])
            #     plt.savefig(f"timestep_{i}.png")
            #     plt.close()

            # Plot VJP fields for each timestep
            # for i in range(batch['vjp'].shape[1]):
            #     temp = SaturationVisualizationCallback().plot_results(batch['vjp'][0][i], batch['vjp'][0][i])
            #     plt.savefig(f"vjp_timestep_{i}.png")
            #     plt.close()

class TestGCSModel(unittest.TestCase):
    def setUp(self):
        self.k_file = "/home/ubuntu/DeFINO/data_generation/src/rescaled_200_fields.h5"
        self.states_dir = "/home/ubuntu/DeFINO/data_generation/src"
        self.vjp_dir = "/home/ubuntu/DeFINO/data_generation/src/num_ev_8"
        self.eigvec_dir = "/home/ubuntu/DeFINO/data_generation/src/num_ev_8"

        data_loader = GCSDataLoader(
            k_file=self.k_file,
            states_dir=self.states_dir,
            batch_size=1,
            nt=5,
            nx=256,
            ny=256,
            num_vec=8,    
            vjp_dir=self.vjp_dir,
            eigvec_dir=self.eigvec_dir,
        )

        self.train_loader = data_loader.get_dataloader(offset=0, limit=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GCSModel()
        self.model.eval()
        self.model.to(self.device)

    def test_model_forward_pass(self):
        for batch in self.train_loader:
            output = self.model(batch['x'].to(self.device))
            self.assertEqual(output.shape, (1, 1, 5, 256, 256))
            break

if __name__ == '__main__':
    unittest.main()
