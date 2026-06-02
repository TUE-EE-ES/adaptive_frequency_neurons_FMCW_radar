import struct
import numpy as np
#import pandas as pd
import io
import os
import sys
import time
from collections import namedtuple

__ver__="20231211"

_tlv_header_t = struct.Struct("<4sL")


def _read_and_parse(file, format):
    n = struct.calcsize(format)
    bytes = file.read(n)
    return struct.unpack(format, bytes)

def _extract_data(stream, dtype, count):
    buffer = stream.read(count * np.dtype(dtype).itemsize)
    if len(buffer) == 0:
        return np.empty((0,), dtype=dtype)
    return np.frombuffer(buffer, dtype=dtype, count=count)

def _extract_uint32(stream, count):
    return _extract_data(stream, np.uint32, count)

def _extract_uint16(stream, count):
    return _extract_data(stream, np.uint16, count)


def create_tlv(tag, value):
    return _tlv_header_t.pack(tag, len(value)) + value


class FrameInfo:
    @staticmethod
    def encoded_format():
        return "<LLLLL"

    @staticmethod
    def encoded_size():
        return struct.calcsize(FrameInfo.encoded_format())

    def __init__(self, header_bytes, radar_type):
        assert len(header_bytes) == self.encoded_size()
        self._num_tx_antennas, \
            self._num_rx_antennas, \
            self._num_chirps_per_frame, \
            self._num_samples_per_chirp, \
            flags = struct.unpack(self.encoded_format(), header_bytes)
        self._with_complex_samples         = ((flags >> 0) & 1) == 1
        self._with_timestamps_per_chirp    = ((flags >> 1) & 1) == 1
        self._with_hw_timestamps_per_chirp = ((flags >> 2) & 1) == 1
        self._with_ant_config_per_chirp    = ((flags >> 3) & 1) == 1
        self._radar_type = radar_type

    @property
    def num_tx_antennas(self):
        return self._num_tx_antennas
        
    @property
    def num_rx_antennas(self):
        return self._num_rx_antennas
        
    @property
    def num_chirps_per_frame(self):
        return self._num_chirps_per_frame
        
    @property
    def num_samples_per_chirp(self):
        return self._num_samples_per_chirp

    @property
    def with_complex_samples(self):
        return self._with_complex_samples

    @property
    def with_timestamps_per_chirp(self):
        return self._with_timestamps_per_chirp

    @property
    def with_hw_timestamps_per_chirp(self):
        return self._with_hw_timestamps_per_chirp

    @property
    def with_ant_config_per_chirp(self):
        return self._with_ant_config_per_chirp

    @property
    def num_tbins_per_frame(self):
        return self._num_chirps_per_frame if self._radar_type == 8 else 1


class RadarFrame:
    def __init__(self, value, frame_info, keep_u16=False):
        stream = io.BytesIO(value)
        # Read timestamps
        n = frame_info.num_chirps_per_frame if frame_info.with_timestamps_per_chirp else 1
        ts_sec = _extract_uint32(stream, n)
        ts_nsec = _extract_uint32(stream, n)
        timestamps = ts_sec + ts_nsec / 1000000000
        # Read td_matrix
        m = 2 if frame_info.with_complex_samples else 1 
        n = frame_info.num_chirps_per_frame * frame_info.num_tx_antennas * frame_info.num_rx_antennas * frame_info.num_samples_per_chirp
        td_matrix = _extract_uint16(stream, n*m)
        if frame_info.num_tx_antennas * frame_info.num_rx_antennas == 1:
            td_matrix_shape = (frame_info.num_chirps_per_frame, frame_info.num_samples_per_chirp)
        else:
            td_matrix_shape = (frame_info.num_chirps_per_frame, frame_info.num_tx_antennas, frame_info.num_rx_antennas, frame_info.num_samples_per_chirp)
        if frame_info.with_complex_samples:
            if keep_u16:
                td_matrix_shape = (*td_matrix_shape, 2)
            else:
                td_matrix = td_matrix[0::2] + 1j * td_matrix[1::2]
        td_matrix = td_matrix.reshape(td_matrix_shape)
        # Read hw_timestamp_pri and hw_timestamp_dma
        if frame_info.with_hw_timestamps_per_chirp:
            hw_timestamp_pri = _extract_uint32(stream, frame_info.num_chirps_per_frame)
            hw_timestamp_dma = _extract_uint32(stream, frame_info.num_chirps_per_frame)
        else:
            hw_timestamp_pri = np.empty((0,))
            hw_timestamp_dma = np.empty((0,))
        # Read ant_config_from_hw
        if frame_info.with_ant_config_per_chirp:
            ant_config_from_hw = _extract_uint16(stream, frame_info.num_chirps_per_frame)
        else:
            ant_config_from_hw = np.empty((0,))
        # All together
        self._td_matrix = td_matrix
        self._timestamps = timestamps
        self._hw_timestamp_pri = hw_timestamp_pri
        self._hw_timestamp_dma = hw_timestamp_dma
        self._ant_config_from_hw = ant_config_from_hw

    @property
    def td_matrix(self):
        return self._td_matrix

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def hw_timestamp_pri(self):
        return self._hw_timestamp_pri

    @property
    def hw_timestamp_dma(self):
        return self._hw_timestamp_dma

    @property
    def ant_config_from_hw(self):
        return self._ant_config_from_hw

def print_offsets(d):
    print("offsets:", [d.fields[name][1] for name in d.names])
    print("itemsize:", d.itemsize)


class PolaritiesBin:
    SIZE_X = 346
    SIZE_Y = 260

    def __init__(self, value):
        d = np.dtype([('timestamp', 'i8'), ('x', 'u2'), ('y', 'u2'), ('polarity', bool)], align=True)
        num_items = len(value) // d.itemsize
        self._polarities = np.frombuffer(value, dtype=d, count=num_items)

    @property
    def polarities(self):
        return self._polarities

    # def mapped(self, width, height, crop_x = 0):
    #     return fast_compute.map_polarities(self._polarities, PolaritiesBin.SIZE_X, PolaritiesBin.SIZE_Y, width, height, crop_x)

    @property
    def size(self):
        return len(self.polarities)


class FileHeader:
    def __init__(self, value):
        head_format = "<HH"
        head_size = struct.calcsize(head_format)
        info_size = FrameInfo.encoded_size()
        header_size = head_size + info_size
        if len(value) < header_size:
            raise Exception("Header too short")
        self._version, self._radar_type = struct.unpack(head_format, value[0:head_size])
        frame_info_bytes = value[head_size:header_size]
        self._frame_info = FrameInfo(frame_info_bytes, self._radar_type)

    @property
    def version(self):
        return self._version

    @property
    def radar_type(self):
        return self._radar_type

    @property
    def frame_info(self):
        return self._frame_info


class RadarFileParser:
    def __init__(self, filepath, read_radar=True, read_dvs=True):
        with open(filepath, 'rb') as file:
            tag, length = self._read_tlv_tag_and_length(file)
            if tag != b'radr':
                raise Exception("No header found")
            header_bytes = self._read_tlv_value(file, length)
            self._header = FileHeader(header_bytes)
            self._frame_info = self._header.frame_info

            radar_frames = []
            dvs_tbins = []
            while True:
                tag, length = self._read_tlv_tag_and_length(file)
                if tag is None:
                    break
                if tag == b"rafd" and read_radar:
                    value = self._read_tlv_value(file, length)
                    f = RadarFrame(value, self._frame_info)
                    radar_frames.append(f)
                elif tag == b"polb" and read_dvs:
                    value = self._read_tlv_value(file, length)
                    t = PolaritiesBin(value)
                    dvs_tbins.append(t)
                else:
                    self._skip_tlv_value(file, length)
            self._radar_frames = radar_frames
            self._dvs_polarity_tbins = dvs_tbins

    @property
    def num_tx_antennas(self):
        return self._frame_info._num_tx_antennas
        
    @property
    def num_rx_antennas(self):
        return self._frame_info.num_rx_antennas
        
    @property
    def num_chirps_per_frame(self):
        return self._frame_info.num_chirps_per_frame
        
    @property
    def num_samples_per_chirp(self):
        return self._frame_info.num_samples_per_chirp

    @property
    def with_complex_samples(self):
        return self._frame_info.with_complex_samples

    @property
    def radar_frames(self):
        return self._radar_frames

    @property
    def num_radar_frames(self):
        return len(self._radar_frames)

    @property
    def radar_chirps(self):
        if len(self._radar_frames) > 0:
            return np.concatenate( list((frame.td_matrix for frame in self._radar_frames)), axis=0 )
        else:
            return self._radar_frames

    @property
    def num_radar_chirps(self):
        return self.num_chirps_per_frame * self.num_radar_frames

    @property
    def dvs_polarity_tbins(self):
        return self._dvs_polarity_tbins

    @property
    def num_dvs_polarity_tbins(self):
        return len(self._dvs_polarity_tbins)

    @staticmethod
    def _read_tlv_tag_and_length(file):
        bytes = file.read(8)
        if len(bytes) < 8:
            return None, "EOF" if len(bytes) == 0 else "Incomplete header"
        tag, length = _tlv_header_t.unpack(bytes)
        return tag, length

    @staticmethod
    def _read_tlv_value(file, length):
        value = file.read(length)
        if len(value) < length:
            return None, "Incomplete value"
        return value

    @staticmethod
    def _skip_tlv_value(file, length):
        file.seek(length, os.SEEK_CUR)


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("specify two args for input and output files")
        exit(0)

    N = 1
    start_time = time.time()
    for i in range(N):
        rf = RadarFileParser(input_file, read_dvs=False)
    print(f"--- {(time.time() - start_time) / N} seconds ---")

    print(len(rf.radar_frames), "frames")

    np.save(output_file, rf.radar_frames)

    from functools import reduce
    print(reduce(lambda x,y: x+y, map((lambda x: len(x.polarities)), rf.dvs_polarity_tbins), 0), "polarities")

else:
    print("\n", __name__, __ver__, ":", time.asctime())
