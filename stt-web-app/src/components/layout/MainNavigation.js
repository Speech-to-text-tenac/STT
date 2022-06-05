import { Link } from "react-router-dom";
import classes from "./MainNavigation.module.css";
// classes in the above can be any name we want

export const MainNavigation = () => {
  return (
    <>
      <div className={classes.nav_contain}>
        <header className={classes.header}>
          <div className={classes.logo}>
            <h1>African language Speech Recognition - Speech-to-Text</h1>
          </div>
        </header>

        <nav className={classes.container}>
          <ul>
            <li>
              <Link to="/">Prediction</Link>
            </li>
            <li>
              <Link to="/spectogram">Spectogram</Link>
            </li>
          </ul>
        </nav>
      </div>
    </>
  );
};
