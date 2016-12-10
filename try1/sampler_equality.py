import random

def gen_equality(m_class, n_objects):
  return [random.randint(1, m_class) for i in range(n_objects)]

def gen_equality_sampler(equality):
  def sampler(n_samples):
    ret = []
    for i in range(n_samples):
      obj1 = random.randint(0, len(equality) - 1)
      obj2 = random.randint(0, len(equality) - 1)
      val1 = equality[obj1]
      val2 = equality[obj2]
      ret.append(((obj1, obj2), 1.0 if val1 == val2 else 0.0))

    return ret
  return sampler
