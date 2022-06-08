import React from "react";
import classes from "./Home.module.css";
export const Home = () => {
  return (
    <div className={classes.content}>
      <p className={classes.para1}>
        Movies and TV shows love to depict robots who can understand and talk
        back to humans. Shows like Westworld, movies like Star Wars and I, Robot
        are filled with such marvels. But what if all of this exists in this day
        and age? Which it certainly does. we can write a program that
        understands what you say and respond to it. All of this is possible with
        the help of speech recognition.
      </p>
      <p className={classes.para2}>
        Speech recognition technology allows for hands-free control of
        smartphones, speakers, and even vehicles in a wide variety of languages.
        Companies have moved towards the goal of enabling machines to understand
        and respond to more and more of our verbalized commands. There are many
        matured speech recognition systems available, such as Google Assistant,
        Amazon Alexa, and Apple’s Siri. However, all of those voice assistants
        work for limited languages only.{" "}
      </p>
      {/* <video className={classes.videobackground} loop muted autoplay>
        <source src={process.env.PUBLIC_URL + "/img/AudioWave.mp4"}></source>
      </video> */}
      <p className={classes.para3}>
        In here you will see a speech reconginition that is been done for World
        Food Program. The World Food Program wants to deploy an intelligent form
        that collects nutritional information of food bought and sold at markets
        in two different countries in Africa - Ethiopia and Kenya. The design of
        this intelligent form requires selected people to install an app on
        their mobile phone, and whenever they buy food, they use their voice to
        activate the app to register the list of items they just bought in their
        own language. The intelligent systems in the app are expected to live to
        transcribe the speech-to-text and organize the information in an
        easy-to-process way in a database.{" "}
      </p>
      <p className={classes.para4}>
        Before we go in to transcribing a speech to text. Let's take a one voice
        record and see what the visualizaition of the voice and it transcription
        is: <br></br>
        <img
          src={process.env.PUBLIC_URL + "/img/upload.png"}
          alt="audio"
        />{" "}
        <br></br>
        for this speech reconginition model we will use an audio with .wav
        format. Okay, let see the visualizaiton of the audio data
      </p>
      <br></br>

      <p className={classes.para5}>
        <img src={process.env.PUBLIC_URL + "/img/11.png"} alt="audio" />{" "}
        <br></br>
        This shows us the differnt frequency amplitudes the sound of the audio
    </p>
      <br></br>
      <p className={classes.para6}>The Transcription of the sound</p>
      <br></br>
      <p className={classes.para7}>ያንደኛ ደረጃ ትምህርታቸው ን ጐንደር ተ ም ረዋል</p>
    </div>
  );
};
