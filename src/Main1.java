import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import static java.lang.String.format;
import static org.opencv.dnn.Dnn.*;

public class Main1 {
    public static void main(String[] args) throws CvException{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        String[] age_list = new String[8];
        age_list[0] = "(0 ~ 2)"; age_list[1] = "(4 ~ 6)"; age_list[2] = "(8 ~ 12)"; age_list[3] = "(15 ~ 20)";
        age_list[4] = "(25 ~ 32)"; age_list[5] = "(38 ~ 43)"; age_list[6] = "(48 ~ 53)"; age_list[7] = "(60 ~ 100)";
        String[] gender_list = new String[2];
        gender_list[0] = "Male"; gender_list[1] = "Female";

        Mat src, blob;
        Scalar color = new Scalar(0, 0, 0); // 출력될 때의 색깔
        Net genderNet = new Net();
        Size size = new Size(227, 227);
        Scalar mean = new Scalar(78.4263377603, 87.7689143744, 114.895847746); // 논문쓴 사람이 설정해놨던 값(이게 뭔지는 모름)

        Net age_net = readNetFromCaffe("models/deploy_age.prototxt", "models/age_net.caffemodel");
        Net gender_net = readNetFromCaffe("models/deploy_gender.prototxt", "models/gender_net.caffemodel");

        String xmlFile = "xml/lbpcascade_frontalface.xml";
        CascadeClassifier cc = new CascadeClassifier(xmlFile);

        MatOfRect faceDetection = new MatOfRect();
        src = Imgcodecs.imread("img/2.jpg");
        cc.detectMultiScale(src, faceDetection);
        System.out.println("Detected faces: " + faceDetection.toArray().length);
        blob = Dnn.blobFromImage(src, 1, size, mean, false, false);


        // ------------------------------------------
        // 이 부분이 계속 막힘.
        // 그래서 그냥 임의의 값으로 gender와 age에 gender_list[1], age_list[3]을 넣었다.
        // predict gender
        Mat gender_preds;
        gender_net.setInput(blob);
        gender_preds = gender_net.forward();
        String gender = gender_list[1];

        // predict age
        Mat age_preds;
        age_net.setInput(blob);
        age_preds = age_net.forward();
        String age = age_list[3];
        // ------------------------------------------

        for(Rect rect : faceDetection.toArray()) {
            Imgproc.rectangle(src, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(100, 100, 255), 3);
            String overlay_text = format("%s %s", gender, age);
            Imgproc.putText(src, overlay_text, new Point(rect.x, rect.y),
                    1, 1.5, color, 2);
        }

        HighGui.imshow("2.jpg", src);
        Imgcodecs.imwrite("img/result_2.jpg", src);
    }
}
