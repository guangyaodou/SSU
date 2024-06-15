import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():

                pretrained_state_dict = pretrained_checkpoint.state_dict()
                finetuned_state_dict = finetuned_checkpoint.state_dict()

                self.vector = {}
                counter = 0
                total = 0
                for key in pretrained_state_dict:
                    total += 1
                    if isinstance(pretrained_state_dict[key], str):
                        counter += 1
                        continue
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
                print(f"Total keys: {total}, string keys: {counter}")

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    # def apply_to(self, pretrained_model, scaling_coef=1.0):
    #     """Apply a task vector to a pretrained model."""
    #     with torch.no_grad():
    #         new_state_dict = {}
    #         pretrained_state_dict = pretrained_model.state_dict()
    #         for key in pretrained_state_dict:
    #             if key not in self.vector:
    #                 print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
    #                 continue
    #
    #             new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
    #
    #     pretrained_model.load_state_dict(new_state_dict, strict=False)
    #     return pretrained_model
    def apply_to(self, pretrained_model, scaling_coef=1.0, chunk_size=100):
        """Apply a task vector to a pretrained model in chunks."""
        with torch.no_grad():
            pretrained_state_dict = pretrained_model.state_dict()
            keys = list(self.vector.keys())  # Get all the parameter keys in the task vector
            total_keys = len(keys)
            for i in range(0, total_keys, chunk_size):
                new_state_dict = {}
                for key in keys[i:i + chunk_size]:
                    if key not in pretrained_state_dict:
                        print(f'Warning: key {key} is present in the task vector but not in the pretrained model')
                        continue
                    # Apply scaling and update the parameter
                    new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]

                # Partially load the updated state dict to the model
                pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


# import torch
#
# class TaskVector():
#     def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, is_data_parallel=False):
#         """Initializes the task vector from a pretrained and a finetuned checkpoints."""
#         if vector is not None:
#             self.vector = vector
#         else:
#             assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
#             with torch.no_grad():
#                 pretrained_state_dict = pretrained_checkpoint.state_dict()
#                 finetuned_state_dict = finetuned_checkpoint.state_dict()
#
#                 self.vector = {}
#                 counter = 0
#                 total = 0
#                 for key in pretrained_state_dict:
#                     total += 1
#                     if isinstance(pretrained_state_dict[key], str):
#                         counter += 1
#                         continue
#                     if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
#                         continue
#
#                     if is_data_parallel:
#                         # Remove the "module." prefix from the key
#                         key = key[7:]
#
#                     self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
#                 print(f"Total keys: {total}, string keys: {counter}")
#
#     def __add__(self, other):
#         """Add two task vectors together."""
#         with torch.no_grad():
#             new_vector = {}
#             for key in self.vector:
#                 if key not in other.vector:
#                     print(f'Warning, key {key} is not present in both task vectors.')
#                     continue
#                 new_vector[key] = self.vector[key] + other.vector[key]
#         return TaskVector(vector=new_vector)
#
#     def __radd__(self, other):
#         if other is None or isinstance(other, int):
#             return self
#         return self.__add__(other)
#
#     def __neg__(self):
#         """Negate a task vector."""
#         with torch.no_grad():
#             new_vector = {}
#             for key in self.vector:
#                 new_vector[key] = - self.vector[key]
#         return TaskVector(vector=new_vector)
#
#     def apply_to(self, pretrained_model, scaling_coef=1.0, is_data_parallel=False):
#         """Apply a task vector to a pretrained model."""
#         with torch.no_grad():
#             new_state_dict = {}
#             pretrained_state_dict = pretrained_model.state_dict()
#             for key in pretrained_state_dict:
#                 if is_data_parallel:
#                     # Remove the "module." prefix from the key
#                     task_vector_key = key[7:]
#                 else:
#                     task_vector_key = key
#
#                 if task_vector_key not in self.vector:
#                     print(
#                         f'Warning: key {task_vector_key} is present in the pretrained state dict but not in the task vector')
#                     continue
#                 new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[task_vector_key]
#         pretrained_model.load_state_dict(new_state_dict, strict=False)
#         return pretrained_model
