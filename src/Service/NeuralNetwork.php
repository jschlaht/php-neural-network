<?php

namespace App\Service;

class NeuralNetwork
{
    private array $weightsInputToHidden;
    private array $weightsHiddenToOutput;

    public function __construct(
        private int $inputNodes,
        private int $hiddenNodes,
        private int $outputNodes,
        private float $learningRate
    )
    {
        $this->weightsInputToHidden = $this->createWeightsMatrix($this->hiddenNodes, $this->inputNodes);
        $this->weightsHiddenToOutput = $this->createWeightsMatrix($this->outputNodes, $this->hiddenNodes);
    }

    public function getInputNodes(): int
    {
        return $this->inputNodes;
    }

    public function getHiddenNodes(): int
    {
        return $this->hiddenNodes;
    }

    public function getOutputNodes(): int
    {
        return $this->outputNodes;
    }

    public function getLearningRate(): float
    {
        return $this->learningRate;
    }

    public function getWeightsInputToHidden(): array
    {
        return $this->weightsInputToHidden;
    }

    public function getWeightsHiddenToOutput(): array
    {
        return $this->weightsHiddenToOutput;
    }

    public function train()
    {

    }

    public function query()
    {

    }

    private function createWeightsMatrix(int $rows, int $cols)
    {
        $matrix = [];

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $matrix[$i][$j] = mt_rand(-1000, 1000)/1000;
            }
        }
        return $matrix;
    }
}
