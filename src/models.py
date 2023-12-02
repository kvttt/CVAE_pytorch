import torch
import torch.nn as nn

from collections.abc import Sequence


class Encoder(nn.Module):
	def __init__(self,
	             input_size: Sequence[int],
	             channels: Sequence[int],
	             hidden_size: int,
	             latent_size: int,
	             ):
		super(Encoder, self).__init__()

		assert len(input_size) == 3, "Input size must be 3D"
		assert len(channels) > 0, "Channels must be a non-empty sequence"

		self.conv_layers = []

		for i, channel in enumerate(channels):
			if i == 0:
				self.conv_layers.append(
					nn.Conv3d(
						1,
						channel,
						kernel_size=3,
						stride=2,
						padding=1,
					)
				)
			else:
				self.conv_layers.append(
					nn.Conv3d(
						channels[i - 1],
						channel,
						kernel_size=3,
						stride=2,
						padding=1,
					)
				)

			self.conv_layers.append(
				nn.ReLU()
			)

		self.conv_layers = nn.Sequential(*self.conv_layers)

		fc_feature_size = input_size[0] * input_size[1] * input_size[2] * channels[-1] / (8 ** len(channels))
		assert fc_feature_size == int(fc_feature_size), "input_size must be divisible by 2^len(channels)"

		self.fc = nn.Linear(
			in_features=int(fc_feature_size),
			out_features=hidden_size,
		)

		self.z_mean = nn.Linear(
			in_features=hidden_size,
			out_features=latent_size,
		)

		self.z_log_var = nn.Linear(
			in_features=hidden_size,
			out_features=latent_size,
		)

	@staticmethod
	def sample(z_mean, z_log_var):
		eps = torch.randn_like(z_mean)
		return z_mean + torch.exp(z_log_var / 2) * eps

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		z_mean = self.z_mean(x)
		z_log_var = self.z_log_var(x)
		z = self.sample(z_mean, z_log_var)


class Decoder(nn.Module):
	def __init__(self,
	             output_size: Sequence[int],
	             channels: Sequence[int],
	             hidden_size: int,
	             latent_size: int,
	             ):
		super(Decoder, self).__init__()
		self.lowest_resolution = [int(output_size[i] / (2 ** len(channels))) for i in range(3)]
		self.channels = channels

		assert len(output_size) == 3, "Output size must be 3D"
		assert len(channels) > 0, "Channels must be a non-empty sequence"

		self.fc0 = nn.Linear(
			in_features=latent_size,
			out_features=hidden_size,
		)

		fc_feature_size = output_size[0] * output_size[1] * output_size[2] * channels[0] / (8 ** len(channels))
		assert fc_feature_size == int(fc_feature_size), "input_size must be divisible by 2^len(channels)"

		self.fc1 = nn.Linear(
			in_features=hidden_size,
			out_features=int(fc_feature_size),
		)

		self.conv_layers = []

		self.conv_layers.append(
			nn.ConvTranspose3d(
				channels[0],
				channels[0],
				kernel_size=3,
				stride=2,
				padding=1,
				output_padding=1,
			)
		)

		self.conv_layers.append(
			nn.ReLU()
		)

		for i, channel in enumerate(channels):
			if i == len(channels) - 1:
				self.conv_layers.append(
					nn.ConvTranspose3d(
						channel,
						1,
						kernel_size=3,
						stride=1,
						padding=1,
					)
				)

				self.conv_layers.append(
					nn.Sigmoid()
				)
			else:
				self.conv_layers.append(
					nn.ConvTranspose3d(
						channel,
						channels[i + 1],
						kernel_size=3,
						stride=2,
						padding=1,
						output_padding=1,
					)
				)

				self.conv_layers.append(
					nn.ReLU()
				)

		self.conv_layers = nn.Sequential(*self.conv_layers)

	def forward(self, z):
		x = self.fc1(self.fc0(z))
		x = x.view(-1, self.channels[0], *self.lowest_resolution)
		x = self.conv_layers(x)
		return x


if __name__ == "__main__":
	from torchinfo import summary

	encoder = Encoder(
		input_size=(64, 64, 64),
		channels=(96, 192),
		hidden_size=128,
		latent_size=32,
	)

	decoder = Decoder(
		output_size=(64, 64, 64),
		channels=(192, 96),
		hidden_size=128,
		latent_size=32,
	)

	summary(encoder, input_size=(1, 1, 64, 64, 64))
	summary(decoder, input_size=(1, 1, 32))

	print(encoder)
	print(decoder)
