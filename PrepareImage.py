from backgroundremover.bg import remove
from PIL import Image

class PrepareImage:
    def __init__(self):
        pass
    
    def remove_bg(self, src_img_path, out_img_path):
        model_choices = ["u2net", "u2net_human_seg", "u2netp"]
        f = open(src_img_path, "rb")
        data = f.read()
        img = remove(data, model_name=model_choices[0],
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_structure_size=10,
                    alpha_matting_base_size=1000)
        f.close()
        f = open(out_img_path, "wb")
        f.write(img)
        f.close()


    def black_out(self, src_img_path, out_img_path):
        im = Image.open(src_img_path)
        alpha = im.getchannel('A')
        alphaThresh = alpha.point(lambda p: 255 if p>200 else 0)
        
        res = Image.new('RGB', im.size)
        res.putalpha(alphaThresh)
        res.save(out_img_path)

    def add_white_background(self, src_img_path, out_img_path):
        im = Image.open(src_img_path)

        new_im = Image.new("RGBA", im.size, (255, 255, 255))

        new_im.paste(im, (0, 0), im)
        new_im.save(out_img_path)