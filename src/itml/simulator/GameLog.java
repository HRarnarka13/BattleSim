package itml.simulator;

import itml.simulator.StateBattle;

import java.util.ArrayList;

/**
 *
 *  This class implements the type GameLog, which is used to record the history of a game
 *  (the sequence of states that occur in a battle).
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class GameLog
{
    private ArrayList<StateBattle> m_states = new ArrayList<StateBattle>();

     /**
     *
     * Empty (clear) the log.
     *
     */
    public void clear()
    {
        m_states.clear();
    }

    /**
     *
     * Add a battle state to the game log.
     *
     * @param  bs     Battle state.
     *
     */
    public void add( StateBattle bs ) {
        m_states.add( bs );
    }

    /**
    *
    * Get the states in the log.
    *
    * @return  <code>ArrayList</code> of <code>StateBattle</code>
    *
    */
    public ArrayList<StateBattle> getLog() {
        return m_states;
    }

}
