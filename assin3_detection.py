import jetson.inference
import jetson.utils

CONFIDENCE_THRESHOLD = 0.7 
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=CONFIDENCE_THRESHOLD)
input_path = "images2.jpeg"
output_path = "images2out02.jpeg"
input_source = jetson.utils.videoSource(input_path)
output_display = jetson.utils.videoOutput(output_path)  
while output_display.IsStreaming():
    img = input_source.Capture()
    if img is None:
        break    
    detections = net.Detect(img)
    print(f"\n--- current threshold: {CONFIDENCE_THRESHOLD}, number of detected objects: {len(detections)} ---")
    for i, detection in enumerate(detections):
        print(f"Object {i + 1}:")
        print(f"  ClassID    : {detection.ClassID}")
        print(f"  Confidence : {detection.Confidence}")
        print(f"  Left       : {detection.Left}")
        print(f"  Top        : {detection.Top}")
        print(f"  Right      : {detection.Right}")
        print(f"  Bottom     : {detection.Bottom}")
        print(f"  Width      : {detection.Width}")
        print(f"  Height     : {detection.Height}")
        print(f"  Area       : {detection.Area}")
        print(f"  Center     : {detection.Center}")
    output_display.Render(img)
    output_display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    

    break


