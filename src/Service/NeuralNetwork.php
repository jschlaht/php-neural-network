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
        $this->weightsInputToHidden = $this->createWeightsMatrixNormalDistribution($this->hiddenNodes, $this->inputNodes);
        $this->weightsHiddenToOutput = $this->createWeightsMatrixNormalDistribution($this->outputNodes, $this->hiddenNodes);
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

    private function createWeightsMatrixNormalDistribution(int $rows, int $cols, $mean = 0, $standard_deviation = 0.5) {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $u1 = mt_rand() / mt_getrandmax();
                $u2 = mt_rand() / mt_getrandmax();
                $z = sqrt(-2 * log($u1)) * cos(2 * pi() * $u2);
                $normalized_z = ($z * $standard_deviation) + $mean;
                $clamped_value = max(-0.99, min(0.99, $normalized_z));
                $matrix[$i][$j] = $clamped_value;
            }
        }
        return $matrix;
    }

}
