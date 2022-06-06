import { ReactMediaRecorder} from "react-media-recorder"
//import { ReactMediaRecorder, useReactMediaRecorder } from "react-media-recorder"
import classes from "./Spectogram.module.css";
//import AudioReactRecorder, { RecordState } from 'audio-react-recorder'

export const Spectogram = () => {
  // const {status, startRecording, stopRecording, mediaBlobUrl} = useReactMediaRecorder({video: true})
  return (
    <ReactMediaRecorder
      audio
      render={({ status, startRecording, stopRecording, mediaBlobUrl }) => (
        <div className={classes.display}>
          <p>{status}</p>
          <button onClick={startRecording}>Start</button>
          <button onClick={stopRecording}>Stop</button>
          <br></br>
          <br></br>
          <audio src={mediaBlobUrl} autoPlay loop controls></audio>
          <section>
            <p>HELLO, EVERYONE</p>
          </section>
        </div>
      )}
    />
  ); 
}

