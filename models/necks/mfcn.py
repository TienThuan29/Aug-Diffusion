import torch
import torch.nn as nn

__all__ = ["MFCN"]


class MFCN(nn.Module):
      def __init__(
            self,
            inplanes,
            outplanes,
            instrides,
            outstrides,
      ):
            super(MFCN, self).__init__()
            assert isinstance(inplanes, list)
            assert isinstance(outplanes, list) and len(outplanes) == 1 
            assert isinstance(outstrides, list) and len(outstrides) == 1
            assert outplanes[0] == sum(inplanes)
            
            self.inplanes = inplanes
            self.outplanes = outplanes
            self.instrides = instrides
            self.outstrides = outstrides
            
            self.scale_factors = [
                  in_stride / outstrides[0] for in_stride in instrides
            ] # for resizing
            
            self.upsample_list = [
                  nn.UpsamplingBilinear2d(scale_factor=scale_factor)
                  for scale_factor in self.scale_factors
            ]     
            
      
      def forward(self, input):
            features = input["features"]
            assert len(self.inplanes) == len(features)
            
            feature_list = []
            # resize and concat
            for i in range(len(features)):
                  upsample = self.upsample_list[i]
                  resized_feature = upsample(features[i])
                  feature_list.append(resized_feature)
                  
            feature_align = torch.cat(feature_list, dim=1)
            return {
                  "feature_align": feature_align, "outplane": self.get_outplanes()
            }

      def get_outplanes(self):
            return self.outplanes
      
      def get_outstrides(self):
            return self.outstrides

"""
MFCN nhận một list feature nhiều tỉ lệ, đưa tất cả về cùng kích thước rồi concat theo chiều kênh.
- Từ config: outstrides: [16]
- MFCN sẽ resize tất cả features về cùng stride=16
"""


