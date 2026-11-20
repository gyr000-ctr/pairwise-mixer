"""mixer 核心 API（初始未实现）。"""


class Mixer:
    def mix(self, tracks, gains):
        """混音"""
        raise NotImplementedError

    def peak(self):
        """峰值"""
        raise NotImplementedError

    def normalize(self, samples):
        """归一化"""
        raise NotImplementedError
