import csv
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def save_pictures(axis_x, axis_y, line_color, line_label, name):
    fig, ax = plt.subplots()
    ax.plot(axis_x, axis_y, color=line_color, label=line_label)
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.savefig(name)


def get_prefix():
    f = open('prefix.txt', 'r')
    num_start = int(f.read())
    f.close()
    f = open('prefix.txt', 'w')
    f.write(str(num_start + 1))
    f.close()

    return str(num_start).zfill(4)


def regulate_learning_rate(optimizer, epoch, frequence):
    if not epoch % frequence and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer
